"""Reconstruct the documented historical protocol and audit image-byte exposure."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'code/cmllm/scripts'))
from build_mr_ref_counterfactual_train import read_jsonl, write_jsonl, split_bucket, build_counterfactual_episode
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
data = root / 'shared/data/final_accepted_v1'
out = root / 'research_log/cycle023'
out.mkdir(parents=True, exist_ok=True)
sources = {name: data / name for name in ['episodes_miner_to_helmet_clean_train.jsonl',
            'episodes_counterfactual_clean_val.jsonl', 'helmet_miner_pairs_accept_high.jsonl']}
regular = read_jsonl(sources['episodes_miner_to_helmet_clean_train.jsonl'])
cf = read_jsonl(sources['episodes_counterfactual_clean_val.jsonl'])
pairs = {r['pair_id']: r for r in read_jsonl(sources['helmet_miner_pairs_accept_high.jsonl'])}
train_regular = [r for r in regular if split_bucket(r['image_path']) < 8]
train_cf = [r for r in cf if split_bucket(r['image_path']) < 8]
grouped = [build_counterfactual_episode(r, pairs) for r in train_cf]
assert all(r is not None for r in grouped)
reconstructed = root / 'outputs/cycle023/episodes_miner_to_helmet_plus_counterfactual_train_reconstructed.jsonl'
write_jsonl(reconstructed, train_regular + grouped)
images = defaultdict(list)
with zipfile.ZipFile(root / 'shared/source/mining_helmet.zip') as archive:
    member_hashes = {}
    for kind, rows in [('regular', train_regular), ('counterfactual', train_cf)]:
        for row in rows:
            member = row['image_member']
            if member not in member_hashes:
                member_hashes[member] = hashlib.sha256(archive.read(member)).hexdigest()
            images[member_hashes[member]].append({'kind': kind,
                 'row_id': row.get('episode_id', row.get('counterfactual_id')),
                 'image_member': member, 'historical_image_path': row['image_path'],
                 'bucket': split_bucket(row['image_path'])})
selection = json.loads((root / 'research_log/cycle022/selection_receipt.json').read_text())['selected']
val = json.loads((root / 'research_log/cycle006/split_receipt.json').read_text())['selected']['val']
intersections = {}
for name, entries in [('cycle022_confirmation50', selection), ('cycle018_val50', val)]:
    overlaps = [{'group_id': r['group_id'], 'image_sha256': r['image_sha256'],
                 'training_rows': images[r['image_sha256']]} for r in entries if r['image_sha256'] in images]
    intersections[name] = {'evaluation_images': len(entries), 'overlap_images': len(overlaps),
                           'overlaps': overlaps}
registry = [{'image_sha256': h, 'training_rows': rows} for h, rows in sorted(images.items())]
(out / 'reconstructed_training_image_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
receipt = {'status': 'UNKNOWN',
    'status_reason': 'Exact historical w15 run-specific manifest/hash and environment overrides unavailable; reconstructed protocol is evidence, not a cryptographically bound training manifest.',
    'checkpoint_model_files': json.loads((root / 'research_log/cycle021/inference_provenance.json').read_text())['model_files_sha256'],
    'exact_historical_training_manifest': None, 'exact_training_unique_image_count': None,
    'exact_training_intersections': {'cycle022_confirmation50': None, 'cycle018_val50': None},
    'reconstructed_manifest': {'path': str(reconstructed), 'sha256': sha(reconstructed),
        'regular_rows': len(train_regular), 'counterfactual_rows': len(grouped),
        'total_rows': len(train_regular) + len(grouped), 'unique_image_bytes': len(images),
        'matches_historical_log_counts': (len(train_regular), len(grouped)) == (9724, 3731),
        'source_hashes': {str(p): sha(p) for p in sources.values()},
        'builder_sha256': sha(root / 'code/cmllm/scripts/build_mr_ref_counterfactual_train.py'),
        'regular_input': str(sources['episodes_miner_to_helmet_clean_train.jsonl']),
        'bucket_rule': 'md5(image_path) first8hex modulo10 in0..7; paths may alias identical image bytes'},
    'reconstructed_protocol_intersections': intersections,
    'historical_evaluation_exposure': 'May16 PROJECT_LOG reports full921 counterfactual holdout evaluation and checkpoint comparison before w15 selection.',
    'new_model_forwards': 0}
(out / 'protocol_reconstruction_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': receipt['status'], 'counts': receipt['reconstructed_manifest'],
                  'intersections': {k: v['overlap_images'] for k, v in intersections.items()}}, indent=2))
