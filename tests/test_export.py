import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import data as h
import tempfile,unittest,base64,json
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
class ExportTests(unittest.TestCase):
 def setUp(self):
  self.schema=pa.schema([('id',pa.int64()),('nested',pa.struct([('bytes',pa.binary()),('path',pa.string())])),('list',pa.list_(pa.binary()))],metadata={b'original':b'metadata'})
  self.rows=[{'id':1,'nested':{'bytes':b'RIFF0000WAVEoriginal','path':'../../same.wav'},'list':[b'abc',None,b'']},{'id':2,'nested':None,'list':[]}]
  self.group=type('Group',(),{'attrs':{'arrow_schema_base64':base64.b64encode(self.schema.serialize()).decode()}})()
 def test_both_formats_roundtrip_schema_nulls_nested_media(self):
  for fmt in ['parquet','json']:
   with self.subTest(fmt=fmt),tempfile.TemporaryDirectory() as tmp,patch('examples.heartbeat_access.rows_from_shard',side_effect=lambda g,stop=None:iter(self.rows[:stop])):
    target=Path(tmp)/fmt
    self.assertEqual(h.export_shard(self.group,target,fmt),2)
    if fmt=='parquet':
     table=pq.read_table(target/'data.parquet');self.assertEqual(table.to_pylist(),self.rows);self.assertTrue(table.schema.equals(self.schema,check_metadata=True))
    else:
     data=[json.loads(x) for x in (target/'records.jsonl').read_text().splitlines()]
     self.assertEqual([{f.name:h.decode_json(row[f.name],f.type,target) for f in self.schema} for row in data],self.rows)
    with self.assertRaises(ValueError):h.export_shard(self.group,target,fmt)
 def test_corrupt_media_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'media').mkdir();v=h.encode_json(b'original',pa.binary(),root/'media')
   (root/v['file']).write_bytes(b'changed')
   with self.assertRaises(ValueError):h.decode_json(v,pa.binary(),root)
 def test_range_budget_before_network(self):
  store=h.RangeStore({'x':['https://example.invalid',0,100]},99)
  with self.assertRaises(ValueError):store['x']
 def test_media_path_escape_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):h.decode_json({'file':'../../outside'},pa.binary(),Path(tmp))
