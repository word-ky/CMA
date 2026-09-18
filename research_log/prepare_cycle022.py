"""Conservative byte-level prior-use registry and deterministic holdout selection."""
import hashlib
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'code/cmllm/scripts'))
from build_mr_ref_counterfactual_train import read_jsonl, write_jsonl, split_bucket

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out = root / 'research_log/cycle022'
out.mkdir(exist_ok=True)
source = root / 'shared/data/final_accepted_v1/episodes_counterfactual_clean_val.jsonl'
pool = read_jsonl(source)
pairs = {r['pair_id']: r for r in read_jsonl(root / 'shared/data/final_accepted_v1/helmet_miner_pairs_accept_high.jsonl')}
gid_hash, pair_hash, name_hash, source_bytes = {}, defaultdict(set), defaultdict(set), {}
with zipfile.ZipFile(root / 'shared/source/mining_helmet.zip') as archive:
    for row in pool:
        member = row['image_member']
        if member not in source_bytes:
            source_bytes[member] = hashlib.sha256(archive.read(member)).hexdigest()
        digest = source_bytes[member]
        gid_hash[row['counterfactual_id']] = digest
        name_hash[Path(member).name].add(digest)
        name_hash[Path(row['image_path']).name].add(digest)
        for pid in row['pair_ids']:
            pair_hash[pid].add(digest)
            if pid in pairs:
                name_hash[Path(pairs[pid]['image_path']).name].add(digest)

known_hashes = set(gid_hash.values())
registry, scanned = defaultdict(list), []
for base in [root / 'research_log', root / 'outputs']:
    for path in sorted(base.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.jsonl', '.md', '.txt', '.log'}:
            continue
        if 'cycle022' in path.parts or path.name.startswith('CYCLE022'):
            continue
        content = path.read_text(encoding='utf-8', errors='replace')
        groups = set(re.findall(r'cf_[0-9a-f]{16}', content))
        identities = set(re.findall(r'pair_[0-9a-f]{16}', content))
        names = set(re.findall(r'[^/\\\s"\x27]+\.(?:jpg|jpeg|png)', content, re.I))
        direct_hashes = set(re.findall(r'\b[0-9a-f]{64}\b', content)) & known_hashes
        hits = set(direct_hashes)
        hits.update(gid_hash[g] for g in groups if g in gid_hash)
        for pid in identities:
            hits.update(pair_hash.get(pid, set()))
        for name in names:
            hits.update(name_hash.get(name, set()))
        source_record = {'path': str(path.relative_to(root)), 'sha256': sha(path),
                         'excluded_source_images': len(hits)}
        scanned.append(source_record)
        for digest in hits:
            registry[digest].append(source_record['path'])

receipt = {'scope': 'All prior research_log and outputs text/JSON manifests, including replay copies; current cycle excluded',
           'matching': 'Original archive bytes resolved from group IDs, pair IDs, image basenames, or explicit image SHA256. Conservative inclusion of mentioned/skipped IDs.',
           'pool_sha256': sha(source), 'scanned_files': scanned,
           'unique_used_image_hashes': len(registry),
           'images': [{'sha256': h, 'sources': sorted(set(paths))} for h, paths in sorted(registry.items())]}
(out / 'used_image_registry.json').write_text(json.dumps(receipt, indent=2) + '\n')
holdout = [r for r in pool if split_bucket(r['image_path']) >= 8]
ordered = sorted(holdout, key=lambda r: (hashlib.sha256(('CYCLE022:' + r['counterfactual_id']).encode()).hexdigest(), r['counterfactual_id']))
selected, candidates, seen, excluded = [], [], set(registry), defaultdict(int)
for row in ordered:
    digest = gid_hash[row['counterfactual_id']]
    if digest in registry:
        excluded['previously_used_image_bytes'] += 1
        continue
    ids = row['pair_ids']
    if len(ids) != 2 or len(set(ids)) != 2 or not all(p in pairs for p in ids):
        excluded['invalid_pair_metadata'] += 1
        continue
    if len(set(row['pseudo_miner_ids'])) != 2 or len(set(row['helmet_ids'])) != 2:
        excluded['nondistinct_identity_metadata'] += 1
        continue
    if digest in seen:
        excluded['duplicate_candidate_image_bytes'] += 1
        continue
    seen.add(digest)
    entry = {'group_id': row['counterfactual_id'], 'image_sha256': digest,
             'image_member': row['image_member'], 'pair_ids': ids,
             'order_key': hashlib.sha256(('CYCLE022:' + row['counterfactual_id']).encode()).hexdigest()}
    candidates.append(entry)
    if len(selected) < 50:
        selected.append(row)
write_jsonl(out / 'selected_source_groups.jsonl', selected)
selection = {'status': 'STOP_FEWER_THAN30' if len(selected) < 30 else 'SELECTED_BEFORE_ASSET_RESTORATION',
             'groups': len(selected), 'unused_valid_unique_candidates': len(candidates),
             'holdout_groups': len(holdout), 'registry_sha256': sha(out / 'used_image_registry.json'),
             'source_sha256': sha(source), 'selection_sha256': sha(out / 'selected_source_groups.jsonl'),
             'zero_image_byte_overlap': all(e['image_sha256'] not in registry for e in candidates[:50]),
             'rule': 'sha256(CYCLE022:+counterfactual_id), then ID; first50; original holdout buckets8/9 only',
             'selected': candidates[:50], 'excluded_counts': dict(excluded),
             'mask_validity': 'metadata checked here; restored masks/hashes frozen before inference',
             'model_inference_runs': 0}
(out / 'selection_receipt.json').write_text(json.dumps(selection, indent=2) + '\n')
print(json.dumps({k:v for k,v in selection.items() if k != 'selected'}, indent=2))
