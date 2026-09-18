import fs from 'node:fs';
import crypto from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';
const root='C:/work/CMA/',dir=root+'research_log/cycle026/';
const headers=['cycle','evidence role','method','condition','mIoU','CMSA numerator','CMSA denominator','CMSA rate','Fidelity numerator','Fidelity denominator','Fidelity rate','IER numerator','IER denominator','IER rate','mean margin','median margin','N groups','N identities'];
const keys=['target_miou','cmsa_count','num_groups','cmsa','fidelity_count','num_references','memory_fidelity','ier_count','num_references','identity_error_rate','mean_identity_margin','median_identity_margin','num_groups','num_references'];
const roles={'025':'PRIMARY: DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION','022':'Historical replication: documented overlap','018':'Development replication: documented overlap'};
const rows=[],references=[];let md='# Paper Layer-1 evidence\n\nCycle025 is primary. Cycles022/018 are separately labelled context; no pooling. CSV stores unrounded metric fractions; Markdown displays rates as percentages. Exact w15/ancestor exposure is UNKNOWN for all sets. Targets are reconstructed pseudo labels and memories are supplied. Comparison is system-level.\n';
for(const cycle of ['025','022','018']){
 const path=`research_log/cycle${cycle}/scoring/comparison.json`,buf=fs.readFileSync(root+path),data=JSON.parse(buf);
 md+=`\n## ${roles[cycle]} — Cycle${cycle}\n\n`;
 md+=cycle==='025'?'Selected before inference, disjoint by image SHA256 from the documented training/evaluation registry.23review groups retain incomplete clean-episode QC;27are accepted-pair combinations. Not exact training-unseen.\n\n':`Reconstructed training overlap: ${cycle==='022'?'37/50':'50/50'} images. Source holdout historically evaluated; ${cycle==='022'?'fresh only relative to recorded Cycles001–021':'development-used'}. Not a training-unseen test.\n\n`;
 md+='|Method|Condition|mIoU|CMSA|Fidelity|IER|Mean margin|Median margin|Groups / identities|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n';
 for(const m of ['cma','segllm'])for(const c of ['clean','target15_b']){
  const key=(cycle==='018'&&m==='cma'?'cma_base_w15':m)+'_'+c,s=data.summaries[key],name=m==='cma'?'CMA base-w15':'SegLLM pinned';
  rows.push([cycle,roles[cycle],name,c,...keys.map(k=>s[k])]);
  references.push({csv_row:rows.length+1,path,sha256:crypto.createHash('sha256').update(buf).digest('hex'),pointer:'/summaries/'+key});
  md+=`|${name}|${c}|${(100*s.target_miou).toFixed(2)}%|${s.cmsa_count}/${s.num_groups} (${(100*s.cmsa).toFixed(0)}%)|${s.fidelity_count}/${s.num_references} (${(100*s.memory_fidelity).toFixed(0)}%)|${s.ier_count}/${s.num_references} (${(100*s.identity_error_rate).toFixed(0)}%)|${s.mean_identity_margin.toFixed(6)}|${s.median_identity_margin.toFixed(6)}|${s.num_groups} / ${s.num_references}|\n`;
 }
}
const primary=JSON.parse(fs.readFileSync(root+'research_log/cycle025/scoring/comparison.json'));
for(const c of ['clean','target15_b']){const d=primary.method_deltas[c].aggregate;rows.push(['025','PRIMARY paired delta (fraction units)','CMA minus SegLLM',c,d.target_miou,'','',d.cmsa,'','',d.memory_fidelity,'','',d.identity_error_rate,d.mean_identity_margin,d.median_identity_margin,50,100]);}
const wb=Workbook.create(),sheet=wb.worksheets.add('Paper evidence');sheet.getRange('A1:R15').values=[headers,...rows];wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:'Paper evidence',range:'A1:R13',maxChars:700,tableMaxRows:3,tableMaxCols:8})).ndjson);
const matrix=sheet.getRange('A1:R15').values;
fs.writeFileSync(dir+'PAPER_LAYER1_TABLE.csv',matrix.map(r=>r.map(v=>JSON.stringify(v)).join(',')).join('\n')+'\n');
md+='\n## Primary paired deltas only: CMA minus SegLLM\n\nRates in percentage points; identity margins unscaled.\n\n|Condition|mIoU pp|CMSA pp|Fidelity pp|IER pp|Mean margin|Median margin|\n|---|---:|---:|---:|---:|---:|---:|\n';
for(const c of ['clean','target15_b']){const d=primary.method_deltas[c].aggregate;md+='|'+c+'|'+['target_miou','cmsa','memory_fidelity','identity_error_rate'].map(k=>(100*d[k]).toFixed(2)).join('|')+'|'+d.mean_identity_margin.toFixed(6)+'|'+d.median_identity_margin.toFixed(6)+'|\n';}
fs.writeFileSync(dir+'PAPER_LAYER1_TABLE.md',md);
fs.writeFileSync(dir+'table_sources.json',JSON.stringify({references,keys,delta_source:'research_log/cycle025/scoring/comparison.json',delta_pointer:'/method_deltas'},null,2)+'\n');
