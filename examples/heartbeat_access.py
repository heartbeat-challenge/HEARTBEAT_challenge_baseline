"""Lossless HEARTBEAT records in Zarr v2, addressed by a ReferenceFS index.

Media stays in its original encoding. Row JSON uses schema-directed binary
references, so arbitrary user strings/objects cannot collide with the marker.
"""
import base64
import hashlib
import json

import fsspec
import numpy as np
import pyarrow as pa
import zarr


def open_root(config_location, *, chunk_cache_bytes=128*1024*1024):
    fs = fsspec.filesystem('reference', fo=config_location, skip_instance_cache=True)
    if '.zgroup' not in fs.references:
        raise ValueError('Missing Zarr root metadata')
    for key in fs.references:
        if key.endswith('/.zarray'):
            md = json.loads(fs.cat_file(key))
            if len(md['shape']) != 1 or len(md['chunks']) != 1:
                raise ValueError('Expected one-dimensional dense arrays')
            prefix = key[:-len('.zarray')]
            for i in range((md['shape'][0] + md['chunks'][0] - 1)//md['chunks'][0]):
                if prefix + str(i) not in fs.references:
                    raise ValueError('Missing dense chunk reference: ' + prefix + str(i))
    store=fs.get_mapper('')
    if chunk_cache_bytes:
        # Adjacent samples frequently share a media block. Reuse compressed
        # blocks instead of retransferring the same 16 MiB for each small clip.
        store=zarr.LRUStoreCache(store,max_size=chunk_cache_bytes)
    return zarr.open_group(store, mode='r', zarr_version=2), fs


def restore(value, dtype, group):
    if value is None:
        return None
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        i = value['venus_blob_index']
        start, end = map(int, group['media_offsets'][i:i+2])
        return group['media_data'][start:end].tobytes()
    if pa.types.is_struct(dtype):
        return {f.name: restore(value[f.name], f.type, group) for f in dtype}
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return [restore(v, dtype.value_type, group) for v in value]
    return value


def rows_from_shard(group, start=0, stop=None):
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(group.attrs['arrow_schema_base64'])))
    n = group.attrs['num_rows']
    for i in range(start, min(n, n if stop is None else stop)):
        a, b = map(int, group['records_offsets'][i:i+2])
        value = json.loads(group['records_data'][a:b].tobytes())
        yield {f.name: restore(value[f.name], f.type, group) for f in schema}


def generate_rows(config_location, track, task, split, subset=None, max_rows=None, _index_identity=None):
    if max_rows is not None and max_rows <= 0:
        return
    root, fs = open_root(config_location)
    path = f'data/HEARTBEAT_Challenge/{track}/{task}/{split}'
    if subset:
        path += '/' + subset
    group = root[path]
    count = 0
    for key in sorted(group.group_keys()):
        for row in rows_from_shard(group[key]):
            yield row
            count += 1
            if max_rows is not None and count >= max_rows:
                return


def as_pandas(config_location, track, task, split, subset=None, max_rows=None):
    import pandas as pd
    return pd.DataFrame(generate_rows(config_location, track, task, split, subset, max_rows))


def as_datasets(config_location, track, task, split, subset=None, *, streaming=False,
                cache_dir=None, max_rows=None):
    from datasets import Dataset, IterableDataset, Features
    root, fs = open_root(config_location)
    group = root[f'data/HEARTBEAT_Challenge/{track}/{task}/{split}' + ('/'+subset if subset else '')]
    first = group[sorted(group.group_keys())[0]]
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(first.attrs['arrow_schema_base64'])))
    features = Features.from_arrow_schema(schema)
    kwargs = dict(config_location=str(config_location), track=track, task=task,
                  split=split, subset=subset, max_rows=max_rows)
    # In addition to immutable Hub revision paths, distinguish updated local
    # reference indexes at the same filename without invalidating normal reuse.
    kwargs['_index_identity'] = hashlib.sha256(json.dumps(
        fs.references, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    # Fixed commit URLs/local immutable indexes are part of generator identity.
    if streaming:
        return IterableDataset.from_generator(generate_rows, gen_kwargs=kwargs, features=features)
    # Large original video blobs can overflow Arrow's 2 GiB binary-offset limit
    # when accumulated into the default 1,000-record writer batch.
    return Dataset.from_generator(generate_rows, gen_kwargs=kwargs, cache_dir=cache_dir,
                                  features=features, writer_batch_size=1)
