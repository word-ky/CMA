"""Copy frozen Cycle022 entrypoints with path-only substitution; record equivalence."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
mapping={
 'research_log/freeze_cycle022_assets.py':'research_log/freeze_cycle025_assets.py',
 'research_log/score_cycle022.py':'research_log/score_cycle025.py',
 'research_log/audit_cycle022_weights.py':'research_log/audit_cycle025_weights.py',
 'cmllm_remote/scripts/run_cma_confirmation.py':'research_log/run_cma_cycle025.py',
}
receipt=[]
for source,target in mapping.items():
 s=(root/source).read_text(encoding='utf-8'); t=s.replace('cycle022','cycle025').replace('Cycle022','Cycle025')
 (root/target).write_bytes(t.encode())
 assert t.replace('cycle025','cycle022').replace('Cycle025','Cycle022')==s
 receipt.append({'source':source,'target':target,'source_lf_sha256':hashlib.sha256(s.encode()).hexdigest(),'target_sha256':hashlib.sha256(t.encode()).hexdigest(),'only_change':'cycle022/Cycle022 path and documentation substitution','inverse_substitution_equal':True})
(root/'research_log/cycle025/adapter_equivalence.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
print('Four path-only adapters verified; frozen scientific kernels unchanged.')
