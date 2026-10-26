import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_data", ROOT / "download_data.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory, cls.reference = d.load_assets()

    def test_all_nine_and_66_packs(self):
        self.assertEqual(len(d.select_partitions()), 9)
        files = d.selected_files(self.inventory, d.select_partitions())
        self.assertEqual(len(files), 66)
        self.assertEqual(sum(x['bytes'] for x in files.values()), 599826551742)

    def test_each_partition_contains_only_its_own_packs(self):
        for p in d.PARTITIONS:
            track, task, split, *subset = p.split('/')
            selected = d.select_partitions([track], [task], [split], subset)
            self.assertEqual(selected, [p])
            files = d.selected_files(self.inventory, selected)
            self.assertTrue(files)
            self.assertTrue(all(x.startswith(d.PREFIX + p + '/') for x in files))

    def test_task_numbers_apply_to_both_tracks(self):
        parts = d.select_partitions(tasks=['1'])
        self.assertEqual(len(parts), 4)
        self.assertIn('track2/task1_video/dev', parts)
        self.assertNotIn('track2/task2_audio/dev', parts)

    def test_track1_task2_dev_selects_both_subsets(self):
        self.assertEqual(d.select_partitions(['1'], ['2'], ['dev']),
                         ['track1/task2/dev/dynamic', 'track1/task2/dev/static'])

    def test_invalid_combination(self):
        with self.assertRaises(ValueError):
            d.select_partitions(['2'], ['1'], ['train'], ['static'])

    def test_explicit_video_audio_names(self):
        self.assertEqual(d.select_partitions(tasks=['task2_audio'], splits=['dev']),
                         ['track2/task2_audio/dev'])

    def test_multi_select(self):
        self.assertEqual(len(d.select_partitions(['1', '2'], ['1', '2'], ['dev'])), 5)

    def test_offline_index_filters_metadata_and_urls(self):
        p = 'track1/task1/dev'
        with tempfile.TemporaryDirectory() as tmp:
            result = d.filtered_index(self.reference, self.inventory, [p], Path(tmp))
            for key, value in result['refs'].items():
                if isinstance(value, list):
                    self.assertTrue(key.startswith(d.PREFIX + p + '/'))
                    self.assertTrue(value[0].startswith(str(Path(tmp).resolve()) + '/'))
                    self.assertNotIn('https://', value[0])
                else:
                    self.assertNotIn('track2/', key)
            self.assertEqual(result['venus']['row_count'], 1000)
            self.assertEqual(list(result['venus']['partitions']), [p])
            self.assertIn('.zgroup', result['refs'])
            self.assertIn(d.PREFIX + p + '/.zgroup', result['refs'])

    def test_source_index_not_mutated(self):
        before = copy.deepcopy(self.reference)
        d.filtered_index(self.reference, self.inventory, ['track2/task2_audio/dev'], Path('/tmp/test'))
        self.assertEqual(before, self.reference)

    def test_local_paths_reject_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            (Path(tmp) / 'data').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                d.safe_local_path(Path(tmp), 'data/pack.bin')

    def test_remote_metadata_exact_and_wrong_hash(self):
        files = d.selected_files(self.inventory, ['track1/task1/dev'])
        name, item = next(iter(files.items()))
        record = SimpleNamespace(path=name, size=item['bytes'], lfs=SimpleNamespace(sha256=item['sha256']))
        class API:
            def get_paths_info(self, **kwargs):
                self.kwargs = kwargs
                return [record]
        api = API()
        d.validate_remote(api, self.inventory, files)
        self.assertEqual(api.kwargs['revision'], self.inventory['data_commit'])
        record.lfs.sha256 = '0' * 64
        with self.assertRaises(ValueError):
            d.validate_remote(api, self.inventory, files)

    def test_download_skip_corrupt_refetch_and_sha_failure(self):
        payload = b'original encoded media\x00\xff'
        name = d.PREFIX + 'track1/task1/dev/part-00000.bin'
        files = {name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            path = output / name
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            with contextlib.redirect_stdout(io.StringIO()):
                d.download_packs(self.inventory, files, output, download)
                d.download_packs(self.inventory, files, output, download)
                self.assertEqual(len(calls), 1)
                path.write_bytes(b'x' * len(payload))
                d.download_packs(self.inventory, files, output, download)
                self.assertEqual(calls[-1]['expected'], files[name])
                self.assertEqual(path.read_bytes(), payload)
                path.unlink()
                with self.assertRaises(ValueError):
                    d.download_packs(self.inventory, files, output, lambda **_: path.write_bytes(b'bad'))

    def test_dry_run_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'uncreated'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(d.main(['--track', '1', '--task', '1', '--split', 'dev',
                                         '--dry-run', '--output-dir', str(out)]), 0)
            self.assertFalse(out.exists())

    def test_no_selection_and_conflicting_all_rejected(self):
        for args in ([], ['--all', '--track', '1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                d.main(args)
            self.assertEqual(ctx.exception.code, 2)

    def test_real_hub_client_local_http_download_and_cache(self):
        # Exercise the actual Hub download client against a small local origin.
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from huggingface_hub import hf_hub_download
        payload = b'encoded media payload\x00\xff' * 256
        sha = hashlib.sha256(payload).hexdigest()
        counts = {'get': 0, 'ranges': [], 'mode': 'normal'}
        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('ETag', '"' + sha + '"')
                self.send_header('X-Repo-Commit', 'a' * 40)
                self.end_headers()
            def do_GET(self):
                counts['get'] += 1
                start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
                counts['ranges'].append(start)
                if counts['mode'] == 'ignore_range':
                    start = 0
                self.send_response(206 if start else 200)
                self.send_header('Content-Length', str(len(payload) - start))
                if start:
                    reported = start + 1 if counts['mode'] == 'wrong_range' else start
                    self.send_header('Content-Range', f'bytes {reported}-{len(payload)-1}/{len(payload)}')
                self.end_headers()
                self.wfile.write(payload[start:])
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                kwargs = dict(repo_id='test/heartbeat', filename='part.bin', revision='a' * 40,
                              local_dir=tmp, endpoint=f'http://127.0.0.1:{server.server_port}', token=False)
                downloaded = Path(hf_hub_download(**kwargs))
                self.assertEqual(downloaded.read_bytes(), payload)
                self.assertEqual(Path(hf_hub_download(**kwargs)).read_bytes(), payload)
                self.assertEqual(counts['get'], 1)
                # A new invocation must resume persistent bytes at exactly their offset.
                output = Path(tmp) / 'durable'
                output.mkdir()
                partial = output / 'part.bin.incomplete'
                expected = {'bytes': len(payload), 'sha256': sha}
                opts = dict(repo_id='test/heartbeat', repo_type='model', filename='part.bin',
                            revision='a' * 40, local_dir=output, expected=expected,
                            endpoint=kwargs['endpoint'])
                partial.write_bytes(payload[:400])
                d.download_resumable(**opts)
                self.assertEqual(counts['ranges'][-1], 400)
                self.assertEqual((output / 'part.bin').read_bytes(), payload)
                self.assertFalse(partial.exists())
                # Incorrect Content-Range must not be appended or replace a good file.
                counts['mode'] = 'wrong_range'
                partial.write_bytes(payload[:400])
                with self.assertRaises(ValueError):
                    d.download_resumable(**opts)
                self.assertEqual(partial.read_bytes(), payload[:400])
                self.assertEqual((output / 'part.bin').read_bytes(), payload)
                # A server that ignores Range must restart, never duplicate the prefix.
                counts['mode'] = 'ignore_range'
                d.download_resumable(**opts)
                self.assertEqual((output / 'part.bin').read_bytes(), payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_real_reader_offline_roundtrip(self):
        # Tiny genuine Zarr/ReferenceFS pack, including bytes and nested schema.
        import base64
        import pyarrow as pa
        import zarr
        from examples.heartbeat_access import generate_rows
        p = 'track1/task1/dev'
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = zarr.MemoryStore()
            root = zarr.group(store=store)
            shard = root.require_group(d.PREFIX + p + '/00000')
            schema = pa.schema([('sample_id', pa.string()), ('audio', pa.struct([('bytes', pa.binary())]))])
            shard.attrs.update(arrow_schema_base64=base64.b64encode(schema.serialize().to_pybytes()).decode(), num_rows=1)
            record = json.dumps({'sample_id': 'sample-1', 'audio': {'bytes': {'venus_blob_index': 0}}}).encode()
            media = b'\x00original-media\xff'
            import numpy as np
            for name, values in [('records_data', np.frombuffer(record, dtype='u1')),
                                 ('records_offsets', np.array([0, len(record)], dtype='<i8')),
                                 ('media_data', np.frombuffer(media, dtype='u1')),
                                 ('media_offsets', np.array([0, len(media)], dtype='<i8'))]:
                shard.create_dataset(name, data=values, chunks=(max(1, len(values)),))
            pack_name = d.PREFIX + p + '/part-00000.bin'
            pack = d.safe_local_path(tmp, pack_name)
            pack.parent.mkdir(parents=True)
            refs = {}
            url = f"https://huggingface.co/{self.inventory['repo_id']}/resolve/{self.inventory['data_commit']}/{pack_name}"
            with pack.open('wb') as f:
                for key, value in store.items():
                    value = bytes(value)
                    if key.rsplit('/', 1)[-1].startswith('.'):
                        refs[key] = value.decode()
                    else:
                        refs[key] = [url, f.tell(), len(value)]
                        f.write(value)
            remote = {'version': 1, 'refs': refs, 'venus': {'partitions': {p: {'rows': 1}}}}
            local = d.filtered_index(remote, self.inventory, [p], tmp)
            index = tmp / 'config.json'
            d.atomic_json(index, local)
            rows = list(generate_rows(str(index), track='track1', task='task1', split='dev'))
            self.assertEqual(rows, [{'sample_id': 'sample-1', 'audio': {'bytes': media}}])
            # Whole CLI flow: metadata precheck, selected download, indexes, offline reader.
            expected = {'bytes': pack.stat().st_size, 'sha256': d.digest(pack)}
            inventory = dict(self.inventory, files={pack_name: expected})
            record = SimpleNamespace(path=pack_name, size=expected['bytes'],
                                     lfs=SimpleNamespace(sha256=expected['sha256']))
            api = SimpleNamespace(get_paths_info=lambda **_: [record])
            def transfer(**kwargs):
                import shutil
                target = Path(kwargs['local_dir']) / kwargs['filename']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(pack, target)
            destination = tmp / 'selected-download'
            with patch.object(d, 'load_assets', return_value=(inventory, remote)), \
                    patch('huggingface_hub.HfApi', return_value=api), \
                    patch.object(d, 'download_resumable', side_effect=transfer), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(d.main(['--track', '1', '--task', '1', '--split', 'dev',
                                         '--output-dir', str(destination)]), 0)
            # New orchestration: full selected download -> local conversion -> verification.
            import data as h
            import pyarrow.parquet as pq
            for fmt in ('parquet', 'json'):
                converted = tmp / ('full-' + fmt)
                with patch.object(h, 'download', d), patch.object(d, 'load_assets', return_value=(inventory, remote)), patch('huggingface_hub.HfApi', return_value=api), patch.object(d, 'download_resumable', side_effect=transfer), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(h.main(['--track','1','--task','1','--split','dev','--format',fmt,'--output-dir',str(converted)]), 0)
                exported = converted / 'export' / fmt / p / '00000'
                self.assertTrue(json.loads((exported/'export.json').read_text())['round_trip_verified'])
                if fmt == 'parquet': self.assertEqual(pq.read_table(exported/'data.parquet').to_pylist(), rows)
            row = next(generate_rows(str(destination / 'config.json'), 'track1', 'task1', 'dev'))
            self.assertEqual(row, rows[0])
            self.assertTrue(json.loads((destination / 'download-manifest.json').read_text())['verified_sha256'])


if __name__ == '__main__':
    unittest.main()
