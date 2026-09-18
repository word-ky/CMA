"""Collect pre-existing local history references and archive inventory, without model work."""
import hashlib,json,re,tarfile,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old=Path('D:/work/fightccfa-agin/coalminellm')
out=root/'research_log/cycle024'; out.mkdir(exist_ok=True)
records=[]; archives=[]
def collect(label,data):
 text=data.decode('utf-8',errors='replace')
 tokens=set(re.findall(r'(?:cf_|pair_|ep_)[0-9a-f]{16}|\b[0-9a-f]{64}\b|(?<![\w.-])[\w.-]{1,200}\.(?:jpg|jpeg|png)(?![\w.-])',text,re.I))
 records.append({'source':label,'sha256':hashlib.sha256(data).hexdigest(),'tokens':sorted(tokens)})
def files(base):
 for parent,dirs,names in os.walk(base,followlinks=False):
  dirs[:]=[d for d in dirs if d not in {'node_modules','.git','.venv','__pycache__','cycle024'} and not Path(parent,d).is_junction()]
  for n in sorted(names):yield Path(parent,n)
for base in [root/'research_log',root/'outputs',old]:
 for p in files(base):
  if not p.is_file() or 'node_modules' in p.parts or 'cycle024' in p.parts or p.name.startswith(('CYCLE024','collect_cycle024','inventory_cycle024')): continue
  if 'final_accepted_v1' in p.parts: continue # source pools are not evidence of use
  if p.suffix in {'.json','.jsonl','.md','.txt','.log'}:
   collect(str(p),p.read_bytes())
  if base==old and p.name.endswith(('.tgz','.tar.gz','.zip')):
   record={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   if p.name.endswith(('.tgz','.tar.gz')):
    with tarfile.open(p,'r:gz') as t:
     members=t.getmembers(); record['members']=[m.name for m in members if m.isfile()]
     for m in members:
      if m.isfile() and Path(m.name).suffix in {'.json','.jsonl','.md','.txt','.log'} and 'final_accepted_v1/' not in m.name:
       collect(str(p)+'::'+m.name,t.extractfile(m).read())
   archives.append(record)
(out/'local_history_references.json').write_text(json.dumps({'records':records,'archives':archives},separators=(',',':')),encoding='utf-8')
print(json.dumps({'history_files':len(records),'archives':len(archives)}))
