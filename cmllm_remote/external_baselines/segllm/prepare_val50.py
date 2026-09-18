"""Prepare frozen target-free specs with the unchanged Cycle015 image transform."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_inputs import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--specs', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    specs = json.loads(Path(args.specs).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        for path, expected in spec['input_asset_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        group = out / spec['group_id']
        group.mkdir(exist_ok=True)
        spec_path = group / 'inputs.json'
        spec_path.write_text(json.dumps(spec, indent=2) + '\n')
        receipt = prepare(spec_path, group)
        records.append({'group_id': spec['group_id'],
                        'prepared': str(group / 'preparation_receipt.json'),
                        'source_image': spec['source_image'],
                        'source_sha256': spec['input_asset_hashes'][spec['source_image']],
                        'preparation': receipt})
    assert len(records) == 50
    (out / 'execution_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Prepared 50 groups / 200 fixed trials, no helmet-target reads')


if __name__ == '__main__':
    main()
