"""Rebuild the frozen selection from public parquet; no bundled examples."""
import json
import hashlib
import pyarrow.parquet as pq
from config import ROOT, CODE, RAW_DATA, sha

SOURCES = {
    'data_0.1.2_visual_toolbox_v2.parquet': '42992bf5de25e8d766f820fb9730ece275563ba80dd41e3377bf678c9ba2c2c1',
    'data_v0.8_visual_toolbox_v2.parquet': '96fc256e6f73e098c1b586f1c37baad616ecbddf1105bfca71aa07a5dda7da5a',
}

def main():
    selection = json.loads((CODE/'metadata/train_selection.json').read_text())
    assert len(selection) == len({(r['source'], r['row']) for r in selection}) == 4096
    for name, expected in SOURCES.items():
        assert sha(RAW_DATA/name) == expected, f'Source hash mismatch: {name}'
    out = ROOT/'data'
    out.mkdir(exist_ok=False)
    records = {}
    for source_index, name in enumerate(SOURCES):
        wanted = {r['row']: r for r in selection if r['source'] == name}
        assert len(wanted) == 2048
        offset = 0
        for batch in pq.ParquetFile(RAW_DATA/name).iter_batches(batch_size=64):
            indexes = [i for i in range(batch.num_rows) if offset+i in wanted]
            for i, row in zip(indexes, batch.take(indexes).to_pylist()):
                index = offset+i
                assert row['extra_info']['split'] == 'train' and len(row['images']) == 1
                assert row['env_name'] == 'visual_toolbox_v2'
                content = row['images'][0]['bytes']
                assert hashlib.sha256(content).hexdigest() == wanted[index]['image_sha256']
                image = out/f'{source_index}_{index}.img'
                image.write_bytes(content)
                user = [m['content'] for m in row['prompt'] if m['role'] == 'user']
                assert len(user) == 1
                records[(name,index)] = dict(id=f'{name}:{index}',source=name,source_row=index,
                    category=row['data_source'],question=row['extra_info']['question'],
                    user=user[0].replace('<image>','').strip(),ground_truth=str(row['reward_model']['ground_truth']),
                    image=str(image),image_sha256=wanted[index]['image_sha256'])
            offset += batch.num_rows
    assert len(records) == 4096
    with (out/'train.jsonl').open('x') as f:
        for item in selection:
            f.write(json.dumps(records[(item['source'],item['row'])])+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(passed=True,count=4096,sources=SOURCES,
        selection_sha256=sha(CODE/'metadata/train_selection.json'),train_sha256=sha(out/'train.jsonl')),indent=2))
    print('DATA_READY',out,4096)

if __name__ == '__main__':
    main()
