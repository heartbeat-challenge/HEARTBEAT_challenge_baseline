"""Restore one exported JSON shard using its original Arrow schema."""
import argparse,json,os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from data import decode_json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input-dir',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();folder=args.input_dir.resolve();target=args.output.resolve()
 if target.exists():raise ValueError('Output already exists')
 schema=pa.ipc.read_schema(pa.BufferReader((folder/'schema.arrow').read_bytes()))
 target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.incomplete')
 if temp.exists():raise ValueError('Incomplete output exists')
 with pq.ParquetWriter(temp,schema,compression='zstd',use_compliant_nested_type=False) as writer,(folder/'records.jsonl').open() as stream:
  for line in stream:
   value=json.loads(line);row={f.name:decode_json(value[f.name],f.type,folder) for f in schema}
   writer.write_table(pa.Table.from_pylist([row],schema=schema),row_group_size=1)
 os.replace(temp,target)
 print(target)
if __name__=='__main__':main()
