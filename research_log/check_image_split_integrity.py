"""Check source-image byte SHA256 exclusivity across nominal dataset roles."""
import argparse,json
from collections import defaultdict
from pathlib import Path

def audit_roles(rows):
 roles=defaultdict(set)
 for row in rows: roles[row['image_sha256']].add(row['role'])
 overlaps={h:sorted(r) for h,r in sorted(roles.items()) if len(r)>1}
 return {'status':'FAIL' if overlaps else 'PASS','unique_images':len(roles),'cross_role_images':len(overlaps),'overlaps':overlaps}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--output',required=True);a=p.parse_args()
 result=audit_roles(json.loads(Path(a.manifest).read_text()))
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='overlaps'}))
 raise SystemExit(1 if result['overlaps'] else 0)
