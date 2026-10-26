#!/usr/bin/env python3
"""Select, download and export HEARTBEAT data."""
import argparse
import base64
from collections.abc import MutableMapping
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import download_data as download

class RangeStore(MutableMapping):
    """Bounded sample reads; validate every HTTP range before decoding."""
    def __init__(self, refs, budget):
        self.refs, self.budget, self.received = refs, budget, 0
        self.cache = {}
    def __iter__(self): return iter(self.refs)
    def __len__(self): return len(self.refs)
    def __setitem__(self,k,v): raise TypeError('Read only')
    def __delitem__(self,k): raise TypeError('Read only')
    def __getitem__(self,k):
        import requests
        value=self.refs[k]
        if isinstance(value,str): return value.encode()
        if k in self.cache: return self.cache[k]
        url,offset,length=value
        if self.received+length>self.budget: raise ValueError('Sample download budget exceeded; increase --max-download-mib')
        for attempt in range(4):
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-{offset+length-1}','Accept-Encoding':'identity'},stream=True,timeout=(15,60)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-{offset+length-1}/'):
                        raise ValueError('Server did not honor exact HTTP Range')
                    data=r.raw.read(length+1)
                    self.received += len(data)
                    if len(data)!=length: raise ValueError('Short/oversized HTTP Range')
                break
            except requests.RequestException:
                if attempt==3: raise
                time.sleep(2**attempt)
        self.cache[k]=data
        return data

def media_suffix(value):
    if value.startswith(b'RIFF') and value[8:12]==b'WAVE': return '.wav'
    if value[4:8]==b'ftyp': return '.mp4'
    if value.startswith(b'fLaC'): return '.flac'
    if value.startswith(b'OggS'): return '.ogg'
    if value.startswith(b'ID3'): return '.mp3'
    if value.startswith(b'\xff\xd8\xff'): return '.jpg'
    if value.startswith(b'\x89PNG'): return '.png'
    return '.bin'

def encode_json(value, dtype, media_dir):
    import pyarrow as pa
    if value is None: return None
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype) or pa.types.is_fixed_size_binary(dtype):
        digest=hashlib.sha256(value).hexdigest(); name=digest+media_suffix(value)
        target=media_dir/name
        if not target.exists(): target.write_bytes(value)
        return {'file':'media/'+name,'bytes':len(value),'sha256':digest}
    if pa.types.is_struct(dtype): return {f.name:encode_json(value[f.name],f.type,media_dir) for f in dtype}
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return [encode_json(x,dtype.value_type,media_dir) for x in value]
    if pa.types.is_map(dtype): return [[encode_json(k,dtype.key_type,media_dir),encode_json(v,dtype.item_type,media_dir)] for k,v in value]
    if pa.types.is_decimal(dtype): return str(value)
    if pa.types.is_temporal(dtype):
        integer=pa.int32() if pa.types.is_date32(dtype) or pa.types.is_time32(dtype) else pa.int64()
        return {'temporal_integer':pa.scalar(value,type=dtype).cast(integer).as_py()}
    if isinstance(value,float):
        import math
        if not math.isfinite(value): return {'float':str(value)}
    return value

def decode_json(value,dtype,folder):
    import pyarrow as pa
    if value is None:return None
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype) or pa.types.is_fixed_size_binary(dtype):
        path=(folder/value['file']).resolve()
        if not path.is_relative_to(folder.resolve()): raise ValueError('Unsafe media path')
        data=path.read_bytes()
        if len(data)!=value['bytes'] or hashlib.sha256(data).hexdigest()!=value['sha256']: raise ValueError('Media integrity failure')
        return data
    if pa.types.is_struct(dtype): return {f.name:decode_json(value[f.name],f.type,folder) for f in dtype}
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype): return [decode_json(x,dtype.value_type,folder) for x in value]
    if pa.types.is_map(dtype): return [(decode_json(k,dtype.key_type,folder),decode_json(v,dtype.item_type,folder)) for k,v in value]
    if pa.types.is_decimal(dtype):
        from decimal import Decimal
        return Decimal(value)
    if pa.types.is_temporal(dtype):
        integer=pa.int32() if pa.types.is_date32(dtype) or pa.types.is_time32(dtype) else pa.int64()
        return pa.scalar(value['temporal_integer'],type=integer).cast(dtype).as_py()
    if pa.types.is_floating(dtype) and isinstance(value,dict):return float(value['float'])
    return value

def equal_values(a,b):
    import math
    if isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b):return True
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal_values(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal_values(x,y) for x,y in zip(a,b))
    return a==b

def export_shard(group, target, fmt, limit=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from examples.heartbeat_access import rows_from_shard
    schema=pa.ipc.read_schema(pa.BufferReader(base64.b64decode(group.attrs['arrow_schema_base64'])))
    if target.exists():raise ValueError(f'Export already exists: {target}; use a new --output-dir')
    temp=target.with_name(target.name+'.incomplete')
    if temp.exists():raise ValueError(f'Incomplete export exists: {temp}; inspect/remove it or use another output directory')
    temp.mkdir(parents=True); media=temp/'media'
    if fmt=='json':media.mkdir()
    writer=pq.ParquetWriter(temp/'data.parquet',schema,compression='zstd',use_compliant_nested_type=False) if fmt=='parquet' else None
    out=(temp/'records.jsonl').open('w',encoding='utf-8') if fmt=='json' else None
    count=0
    try:
        for row in rows_from_shard(group,stop=limit):
            if writer: writer.write_table(pa.Table.from_pylist([row],schema=schema),row_group_size=1)
            else:out.write(json.dumps({f.name:encode_json(row[f.name],f.type,media) for f in schema},ensure_ascii=False,allow_nan=False)+'\n')
            count+=1
    finally:
        if writer:writer.close()
        if out:out.close()
    (temp/'schema.arrow').write_bytes(schema.serialize().to_pybytes())
    # Read back and compare all fields/media, one row at a time.
    if fmt=='parquet':
        pf=pq.ParquetFile(temp/'data.parquet')
        if not pf.schema_arrow.equals(schema,check_metadata=True):raise ValueError('Schema mismatch')
        restored=(row for batch in pf.iter_batches(batch_size=1) for row in batch.to_pylist())
    else:
        def json_rows():
            with (temp/'records.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    yield {f.name:decode_json(row[f.name],f.type,temp) for f in schema}
        restored=json_rows()
    import itertools
    sentinel=object()
    for original,actual in itertools.zip_longest(rows_from_shard(group,stop=limit),restored,fillvalue=sentinel):
        if original is sentinel or actual is sentinel or not equal_values(original,actual):raise ValueError('Export round-trip mismatch')
    (temp/'export.json').write_text(json.dumps({'rows':count,'format':fmt,'round_trip_verified':True,'sample':limit is not None},indent=2))
    os.replace(temp,target)
    return count

def main(argv=None):
    p=download.parser()
    p.description=__doc__
    p.add_argument('--format',choices=['packed','parquet','json'],default='parquet')
    p.add_argument('--sample-rows',type=int,help='Read only this many rows PER partition using HTTP Range; no full packs')
    p.add_argument('--max-download-mib',type=int,default=128,help='Sample compressed-block download budget (default 128 MiB)')
    args=p.parse_args(argv)
    filters=[]
    for key in ['track','task','split','subset']:
        if getattr(args,key): filters+=['--'+key,*getattr(args,key)]
    if args.all:filters+=['--all']
    if args.list:return download.main(['--list'])
    if not filters or (args.all and len(filters)>1):p.error('Choose filters OR --all')
    if args.sample_rows is not None and (args.sample_rows<=0 or args.format=='packed'):p.error('--sample-rows must be positive and requires parquet/json')
    if args.max_download_mib<=0:p.error('--max-download-mib must be positive')
    if args.dry_run:return download.main(filters+['--dry-run','--output-dir',str(args.output_dir)])
    import zarr
    from filelock import FileLock
    inventory,reference=download.load_assets()
    parts=download.select_partitions(args.track,args.task,args.split,args.subset)
    output=args.output_dir.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'.heartbeat-export.lock'),timeout=0):
        if args.sample_rows:
            from huggingface_hub import HfApi
            download.validate_remote(HfApi(),inventory,download.selected_files(inventory,parts))
            store=RangeStore(download.filtered_index(reference,inventory,parts)['refs'],args.max_download_mib*1024*1024)
            root=zarr.open_group(store,mode='r')
        else:
            download.main(filters+['--output-dir',str(output/'packed')])
            if args.format=='packed':return 0
            from examples.heartbeat_access import open_root
            root,_=open_root(str(output/'packed/config.json'));store=None
        results={}
        for part in parts:
            group=root[download.PREFIX+part];remaining=args.sample_rows
            for key in sorted(group.group_keys()):
                n=export_shard(group[key],output/('sample' if args.sample_rows else 'export')/args.format/part/key,args.format,remaining)
                results[part+'/'+key]=n
                print(f'Verified {part}/{key}: {n} rows ({args.format})',flush=True)
                if remaining is not None:
                    remaining-=n
                    if remaining<=0:break
        download.atomic_json(output/'export-manifest.json',{'data_commit':inventory['data_commit'],'format':args.format,'sample_rows_per_partition':args.sample_rows,'rows':results,'range_download_bytes':store.received if store else 0,'round_trip_verified':True})
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except Exception as e:
        print(f'Failed ({type(e).__name__}): '+(str(e) if isinstance(e,ValueError) else 'check network/dependencies and retry'),file=sys.stderr);sys.exit(1)
