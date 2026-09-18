"""Format verified Cycle026 table values into LaTeX, without scoring."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'research_log/cycle026';out=root/'research_log/cycle027';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((src/'table_verification.json').read_text());assert v['status']=='PASS' and sha(src/'PAPER_LAYER1_TABLE.csv')==v['csv_sha256']
rows=list(csv.DictReader((src/'PAPER_LAYER1_TABLE.csv').open(encoding='utf-8',newline='')))
def row(r):
 rate=lambda k:f"{float(r[k])*100:.2f}"
 count=lambda n,d,k:f"{r[n]}/{r[d]} ({float(r[k])*100:.0f}\\%)"
 return ' & '.join([r['method'],r['condition'].replace('_',r'\_'),rate('mIoU'),count('CMSA numerator','CMSA denominator','CMSA rate'),count('Fidelity numerator','Fidelity denominator','Fidelity rate'),count('IER numerator','IER denominator','IER rate'),f"{float(r['mean margin']):.6f}"])+r' \\'+'\n'
header=r'''\begin{tabular}{llrrrrr}
\toprule
Method & Condition & mIoU (\%) & CMSA & Fidelity & IER & Mean margin \\
\midrule
'''
main=r'''% Requires booktabs; intended for a two-column-width table.
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Primary Cycle025 \texttt{DOCUMENTED\_PROTOCOL\_DISJOINT\_CONFIRMATION}:
50 groups / 100 identity references per condition, with supplied identity memories
and reconstructed pseudo targets. The comparison is system-level; exact run-bound
w15 and ancestor exposure is unknown. Counts and rates are shown for CMSA,
Memory Fidelity and IER; margins are unscaled. The set is disjoint from documented
training/evaluation image-byte registries, not proven exact training-unseen.}
\label{tab:primary-memory}
'''+header+''.join(row(r) for r in rows if r['cycle']=='025' and r['method']!='CMA minus SegLLM')+r'''\bottomrule
\end{tabular}
\end{table*}
'''
supp=r'''% Requires booktabs. These replications must not be pooled with the primary table.
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Separate overlap-affected replications: each block contains 50 groups and
100 identity references per condition. The reconstructed training protocol overlaps
37/50 Cycle022 images and 50/50 Cycle018 images; their source holdout was historically
evaluated. Cycle018 also served development. These are not independent training-unseen
tests. Supplied memories, reconstructed pseudo targets and system-level comparison
limitations apply. No pooling is performed.}
\label{tab:overlap-replications}
'''+header
for cycle,label in [('022','Cycle022: historical replication; 37/50 documented overlap'),('018','Cycle018: development replication; 50/50 documented overlap')]:
 supp+=r'\multicolumn{7}{l}{\textit{'+label+r'}} \\'+'\n'
 supp+=''.join(row(r) for r in rows if r['cycle']==cycle)+(r'\bottomrule' if cycle=='018' else r'\midrule')+'\n'
supp+=r'\end{tabular}'+'\n'+r'\end{table*}'+'\n'
for name,text in [('MAIN_TABLE_LATEX.tex',main),('SUPPLEMENTARY_REPLICATION_TABLE_LATEX.tex',supp)]:
 (out/name).write_bytes(text.encode())
 selected=[r for r in rows if (r['cycle']=='025' if name.startswith('MAIN') else r['cycle'] in ['022','018']) and r['method']!='CMA minus SegLLM']
 assert all(row(r) in text for r in selected)
(out/'table_format_verification.json').write_text(json.dumps({'source_csv_sha256':sha(src/'PAPER_LAYER1_TABLE.csv'),'source_verification_sha256':sha(src/'table_verification.json'),'main_rows':4,'supplementary_rows':8,'each_rendered_row_matches_verified_csv':True,'model_calls':0,'scorer_calls':0,'tex_sha256':{p.name:sha(p) for p in out.glob('*TABLE_LATEX.tex')}},indent=2)+'\n')
print('LaTeX tables: 4 primary rows and 8 separate supplementary rows verified.')
