#!/usr/bin/env python3
"""Download selected HEARTBEAT partitions and build an offline ReferenceFS index."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
PREFIX = "data/HEARTBEAT_Challenge/"
PARTITIONS = (
    "track1/task1/train", "track1/task1/dev", "track1/task2/train",
    "track1/task2/dev/dynamic", "track1/task2/dev/static",
    "track2/task1_video/train", "track2/task1_video/dev",
    "track2/task2_audio/train", "track2/task2_audio/dev",
)


def load_assets():
    inventory = json.loads((HERE / "data_assets/packs.json").read_text())
    reference = json.loads((HERE / "data_assets/reference.json").read_text())
    commit = inventory["data_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Data revision must be an immutable 40-character commit")
    for name, item in inventory["files"].items():
        parts = PurePosixPath(name).parts
        if (not name.startswith(PREFIX) or ".." in parts or "\\" in name
                or name.rsplit("/", 1)[0][len(PREFIX):] not in PARTITIONS
                or not re.fullmatch(r"part-[0-9]{5}\.bin", parts[-1])
                or not isinstance(item["bytes"], int) or item["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError(f"Invalid pack manifest entry: {name}")
    expected = f"/{inventory['repo_id']}/resolve/{commit}/"
    referenced = set()
    for key, value in reference["refs"].items():
        if isinstance(value, str):
            continue
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"Unsupported reference: {key}")
        url, offset, length = value
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
                or parsed.query or parsed.fragment or not parsed.path.startswith(expected)):
            raise ValueError(f"Unexpected data URL for {key}")
        name = unquote(parsed.path[len(expected):])
        if (name not in inventory["files"] or type(offset) is not int
                or type(length) is not int or offset < 0 or length <= 0
                or offset + length > inventory["files"][name]["bytes"]):
            raise ValueError(f"Invalid pack range for {key}")
        partition = name.rsplit("/", 1)[0]
        if not key.startswith(partition + "/"):
            raise ValueError(f"Cross-partition reference: {key}")
        referenced.add(name)
    if referenced != set(inventory["files"]):
        raise ValueError("Pack inventory and ReferenceFS index disagree")
    return inventory, reference


def select_partitions(tracks=None, tasks=None, splits=None, subsets=None):
    tracks = {"track" + x.removeprefix("track") for x in (tracks or ["1", "2"])}
    tasks = {x.removeprefix("task") for x in (tasks or ["1", "2"])}
    splits = set(splits or ["train", "dev"])
    subsets = set(subsets or [])
    selected = []
    for partition in PARTITIONS:
        track, task, split, *rest = partition.split("/")
        task_number = task.removeprefix("task").split("_")[0]
        if track not in tracks or not ({task_number, task.removeprefix("task")} & tasks):
            continue
        if split not in splits or (subsets and (not rest or rest[0] not in subsets)):
            continue
        selected.append(partition)
    if not selected:
        raise ValueError("No matching partition; dynamic/static exist only in track1/task2/dev")
    return selected


def selected_files(inventory, partitions):
    prefixes = tuple(PREFIX + p + "/" for p in partitions)
    return {k: v for k, v in sorted(inventory["files"].items()) if k.startswith(prefixes)}


def filtered_index(reference, inventory, partitions, output=None):
    """Keep only selected partition metadata and its ancestors; rewrite ranges locally."""
    roots = [PREFIX + p for p in partitions]
    ancestors = {".zgroup", ".zattrs"}
    for root in roots:
        parts = root.split("/")
        for i in range(1, len(parts)):
            ancestors.update("/".join(parts[:i]) + "/" + meta for meta in (".zgroup", ".zattrs"))
    refs = {}
    url_prefix = f"https://huggingface.co/{inventory['repo_id']}/resolve/{inventory['data_commit']}/"
    for key, value in reference["refs"].items():
        if key not in ancestors and not any(key.startswith(root + "/") for root in roots):
            continue
        if isinstance(value, list) and output is not None:
            name = unquote(value[0][len(url_prefix):])
            value = [str(safe_local_path(output, name)), value[1], value[2]]
        refs[key] = value
    result = {"version": 1, "refs": refs, "venus": copy.deepcopy(reference.get("venus", {}))}
    all_stats = result["venus"].get("partitions", {})
    result["venus"]["partitions"] = {p: all_stats[p] for p in partitions if p in all_stats}
    result["venus"]["row_count"] = sum(x["rows"] for x in result["venus"]["partitions"].values())
    result["venus"]["selected_partitions"] = list(partitions)
    return result


def safe_local_path(output, name):
    output = Path(output).resolve()
    path = (output / name).resolve()
    if not path.is_relative_to(output):
        raise ValueError(f"Output path escapes destination: {name}")
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def matches(path, expected):
    return path.is_file() and path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"]


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_remote(api, inventory, files):
    remote = api.get_paths_info(
        repo_id=inventory["repo_id"], paths=list(files), repo_type="model",
        revision=inventory["data_commit"],
    )
    found = {x.path: x for x in remote}
    for name, expected in files.items():
        actual = found.get(name)
        if (actual is None or actual.size != expected["bytes"] or not actual.lfs
                or actual.lfs.sha256 != expected["sha256"]):
            raise ValueError(f"Remote size/LFS SHA256 differs from pinned manifest: {name}")


def download_packs(inventory, files, output, download):
    for n, (name, expected) in enumerate(files.items(), 1):
        path = safe_local_path(output, name)
        print(f"[{n}/{len(files)}] {name}", flush=True)
        if matches(path, expected):
            print("  SHA256 OK, already downloaded", flush=True)
            continue
        verified_sha = download(repo_id=inventory["repo_id"], repo_type="model", filename=name,
                                revision=inventory["data_commit"], local_dir=str(output), expected=expected)
        if verified_sha != expected["sha256"] and not matches(path, expected):
            raise ValueError(f"Downloaded file failed size/SHA256 verification: {name}; rerun to refetch")
        print("  SHA256 OK", flush=True)


def download_resumable(*, repo_id, repo_type, filename, revision, local_dir, expected,
                       endpoint=None):
    """Durable HTTP Range resume, independent of Hub cache implementation changes."""
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    from tqdm.auto import tqdm
    output = Path(local_dir)
    path = safe_local_path(output, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.incomplete')
    if partial.is_symlink():
        raise ValueError(f"Refusing a symlink for temporary download: {partial.name}")
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision,
                     repo_type=repo_type, endpoint=endpoint)
    # Requests strips Authorization on cross-host redirects, including signed storage URLs.
    headers = build_hf_headers()
    headers['Accept-Encoding'] = 'identity'
    with requests.Session() as session:
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= expected['bytes']:
                if matches(partial, expected):
                    os.replace(partial, path)
                    return expected['sha256']
                # An unaccepted complete/oversized temporary file cannot be resumed.
                with partial.open('wb'):
                    pass
                offset = 0
            if shutil.disk_usage(output).free < expected['bytes'] - offset:
                raise ValueError(f"Insufficient free space for {filename}")
            request_headers = dict(headers)
            if offset:
                request_headers['Range'] = f'bytes={offset}-'
            try:
                with session.get(url, headers=request_headers, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError(f"Unexpected HTTP content encoding: {filename}")
                    if response.status_code == 206:
                        expected_range = f"bytes {offset}-{expected['bytes']-1}/{expected['bytes']}"
                        if response.headers.get('Content-Range') != expected_range:
                            raise ValueError(f"Incorrect HTTP Content-Range: {filename}")
                    elif response.status_code == 200:
                        offset = 0  # Range ignored: safely restart instead of appending.
                    else:
                        raise ValueError(f"Unexpected HTTP status: {response.status_code}")
                    length = response.headers.get('Content-Length')
                    if length is not None and int(length) != expected['bytes'] - offset:
                        raise ValueError(f"Incorrect HTTP Content-Length: {filename}")
                    if shutil.disk_usage(output).free < expected['bytes'] - offset:
                        raise ValueError(f"Insufficient free space for {filename}")
                    with partial.open('ab' if offset else 'wb') as f, tqdm(
                            total=expected['bytes'], initial=offset, unit='B', unit_scale=True,
                            desc=Path(filename).name) as progress:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                if f.tell() + len(block) > expected['bytes']:
                                    raise ValueError(f"Server sent too many bytes: {filename}")
                                f.write(block)
                                progress.update(len(block))
                        f.flush()
                        os.fsync(f.fileno())
                if partial.stat().st_size != expected['bytes']:
                    raise requests.ConnectionError('Short response body')
                if not matches(partial, expected):
                    raise ValueError(f"SHA256 mismatch: {filename}; rerun to refetch")
                os.replace(partial, path)
                return expected['sha256']
            except requests.RequestException as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                if status is not None and status not in (408, 429) and status < 500:
                    raise
                if attempt == 3:
                    raise
                print(f"  Connection interrupted; retry {attempt + 1}/3 from saved bytes", flush=True)
                time.sleep(2 ** attempt)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python download_data.py --track 1 --task 1 --split dev\n"
               "  python download_data.py --track 2 --split train\n"
               "  python download_data.py --task 2 --split dev\n"
               "  python download_data.py --track 1 --task 2 --split dev --subset static\n"
               "  python download_data.py --all --dry-run")
    p.add_argument("--track", nargs="+", choices=["1", "2", "track1", "track2"])
    p.add_argument("--task", nargs="+", choices=["1", "2", "task1", "task2", "task1_video", "task2_audio"])
    p.add_argument("--split", nargs="+", choices=["train", "dev"])
    p.add_argument("--subset", nargs="+", choices=["dynamic", "static"])
    p.add_argument("--all", action="store_true", help="Download all nine partitions")
    p.add_argument("--output-dir", type=Path, default=HERE / "downloads", help="Destination (default: ./downloads)")
    p.add_argument("--dry-run", action="store_true", help="Print selection and full byte count; no downloads or writes")
    p.add_argument("--list", action="store_true", help="List the nine available partitions, then exit")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.list:
        print("\n".join(PARTITIONS))
        return 0
    if args.all and any((args.track, args.task, args.split, args.subset)):
        p.error("Use --all OR partition filters")
    if not args.all and not any((args.track, args.task, args.split, args.subset)):
        p.error("Choose --all or at least one of --track/--task/--split/--subset (use --dry-run to preview)")
    inventory, reference = load_assets()
    partitions = select_partitions(args.track, args.task, args.split, args.subset)
    files = selected_files(inventory, partitions)
    total = sum(x["bytes"] for x in files.values())
    print(f"Source: {inventory['repo_id']} @ {inventory['data_commit']}")
    print("Partitions:\n  " + "\n  ".join(partitions))
    print(f"Packs: {len(files)}; full selection size: {total:,} bytes ({total / 1e9:.3f} GB)")
    output = args.output_dir.expanduser().resolve()
    print(f"Destination: {output}")
    if args.dry_run:
        for name, info in files.items():
            print(f"  {info['bytes']:>14,}  {name}")
        return 0
    # Dependencies are optional for --list and --dry-run.
    try:
        from huggingface_hub import HfApi
        import requests
        from filelock import FileLock
    except ImportError:
        raise ValueError("Install download dependencies: python -m pip install -r requirements-download.txt") from None
    validate_remote(HfApi(), inventory, files)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.heartbeat-download.lock'), timeout=0):
        download_packs(inventory, files, output, download_resumable)
        # Write a usable index only after every selected pack passes full SHA256.
        atomic_json(output / "config.remote.json", filtered_index(reference, inventory, partitions))
        atomic_json(output / "config.json", filtered_index(reference, inventory, partitions, output))
        atomic_json(output / "download-manifest.json", {
            "repo_id": inventory["repo_id"], "data_commit": inventory["data_commit"],
            "partitions": partitions, "files": files, "verified_sha256": True,
        })
    print(f"Complete. Offline index: {output / 'config.json'}")
    print("Repeated runs resume downloads and verify existing files. Other partitions' files are retained.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        # Do not print response/request objects: signed URLs or credentials may be present.
        if isinstance(exc, ValueError):
            print(f"Error: {exc}", file=sys.stderr)
        else:
            print(f"Download stopped ({type(exc).__name__}). Rerun the same command to resume.", file=sys.stderr)
        sys.exit(1)
