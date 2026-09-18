import fs from 'node:fs';
import crypto from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';
const dir='C:/work/CMA/research_log/cycle023/';
const headers=['method','split role','condition','mIoU','CMSA','Fidelity','IER','mean margin','median margin','N groups','N identity predictions'];
const metrics=['target_miou','cmsa','memory_fidelity','identity_error_rate','mean_identity_margin','median_identity_margin','num_groups','num_references'];
const rows=[], sources=[];
for(const cycle of ['022','018']){
 const path=`C:/work/CMA/research_log/cycle${cycle}/scoring/comparison.json`;
 const bytes=fs.readFileSync(path), data=JSON.parse(bytes).summaries;
 for(const method of ['cma','segllm'])for(const condition of ['clean','target15_b']){
  const key=(cycle==='018'&&method==='cma'?'cma_base_w15':method)+'_'+condition;
  rows.push([method==='cma'?'CMA base-w15':'SegLLM pinned',cycle==='022'?'Cycle022 primary confirmation':'Cycle018 development/replication',condition,...metrics.map(m=>data[key][m])]);
  sources.push({row:rows.length+1,path:path.replace('C:/work/CMA/',''),sha256:crypto.createHash('sha256').update(bytes).digest('hex'),pointer:'/summaries/'+key});
 }
}
const wb=Workbook.create(), sheet=wb.worksheets.add('Layer1');
sheet.getRange('A1:K9').values=[headers,...rows]; wb.recalculate();
console.log(await wb.inspect({kind:'region',sheetId:'Layer1',range:'A1:K9',maxChars:1500,tableMaxRows:9,tableMaxCols:11}));
// CSV has no presentation formatting; serialize authored worksheet values without rounding.
const matrix=sheet.getRange('A1:K9').values;
fs.writeFileSync(dir+'MAIN_LAYER1_TABLE.csv',matrix.map(r=>r.map(v=>typeof v==='string'?JSON.stringify(v):String(v)).join(',')).join('\n')+'\n');
const note='Rates in this Markdown table are percentages; CSV stores fractions. Margins are unscaled. Each split remains separate; no pooled result or selective MCR metric. Sources: frozen Cycle022 and Cycle018 scoring/comparison.json, with row pointers and SHA256 in table_provenance.json.\n\n**Provenance limitation:** Cycle022 is untouched only by recorded Cycles001–021 iteration history. The historical full holdout was evaluated during checkpoint comparison. Exact w15 training exposure is UNKNOWN; reconstructed documented training overlaps 37/50 Cycle022 and 50/50 Cycle018 images by SHA256. Neither split establishes training-unseen generalization. Targets are reconstructed pseudo labels; identity memories are supplied. SegLLM is a system-level released-baseline comparison with different training exposure.\n';
let md='# Frozen Layer-1 results\n\n'+note+'\n|'+headers.join('|')+'|\n|'+headers.map(()=> '---').join('|')+'|\n';
for(const r of rows)md+='|'+r.map((v,i)=>i>=3&&i<=6?(100*v).toFixed(2)+'%':i===7||i===8?v.toFixed(6):v).join('|')+'|\n';
fs.writeFileSync(dir+'MAIN_LAYER1_TABLE.md',md);
fs.writeFileSync(dir+'table_provenance.json',JSON.stringify({sources,metrics,rows:8,performance_recomputed:false},null,2)+'\n');
