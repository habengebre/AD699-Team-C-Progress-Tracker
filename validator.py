#!/usr/bin/env python3
"""AD699 Team C notebook validator. Python 3.9+. Standard-library static checking.
For --execute install: pip install nbformat nbclient ipykernel
Execute only trusted Team C notebooks in an isolated environment.
"""
import argparse, datetime as dt, html, json, re, sys, traceback
from pathlib import Path

SECTIONS = [
 ('preparation','Data Preparation & Exploration'),
 ('regression','Multiple Regression'),
 ('knn','K-Nearest Neighbors'),
 ('tree','Classification Tree'),
 ('transformer','Transformer Classification'),
 ('clustering','Clustering'),
 ('conclusion','Conclusions & Submission'),
]
# Each rule is a lightweight indication, NOT proof of statistical correctness.
RULES = {
 'preparation':[
  ('missing','Missing-value investigation',r'(?i)(isna\(|isnull\(|missingno|dropna\(|fillna\()','code'),
  ('summary','Neighborhood summary statistics',r'(?i)(groupby\(|pivot_table\(|\.agg\()','code'),
  ('plots','At least five distinct plot types (human review)',r'(?i)(histplot|hist\(|boxplot|barplot|bar\(|scatterplot|scatter\(|violinplot|violin\(|heatmap|lineplot|line\(|countplot|pie\(|sns\.)','code'),
  ('map','Geographic rental map',r'(?i)(folium|px\.scatter_map|scatter_mapbox|plotly\.express\.scatter_map|geopandas|leaflet)','code'),
  ('wordcloud','Neighborhood word cloud',r'(?i)(WordCloud\(|wordcloud)','code'),
  ('interpret','Written interpretation',r'(?i)(finding|interpret|takeaway|neighbou?rhood|observation)','markdown'),
 ],
 'regression':[
  ('price','Price used as target',r'(?i)(\[\s*[\'\"]price[\'\"]\s*\]|log\s*\(\s*.*price|\blog_price\b)','code'),
  ('fit','Multiple regression estimator',r'(?i)(OLS\(|LinearRegression\(|smf\.ols\(|\.ols\()','code'),
  ('metrics','Regression evaluation',r'(?i)(r2_score|mean_squared_error|root_mean_squared_error|mean_absolute_error|\.rsquared|\.summary\()','code'),
  ('log','Log-price consideration',r'(?i)(log1p\(|np\.log\(|log.price|log_price)','code'),
  ('explain','Coefficient and model interpretation',r'(?i)(coefficient|regression equation|adjusted r|residual|predictor)','markdown'),
 ],
 'knn':[
  ('target','Amenity target preparation',r'(?i)(amenit|wifi|parking|pool|kitchen|air.condition)','code'),
  ('model','KNN classifier',r'KNeighborsClassifier\s*\(','code'),
  ('scale','Feature scaling',r'(StandardScaler|MinMaxScaler|RobustScaler|Pipeline)\s*\(','code'),
  ('tune','k tuning / cross-validation',r'(?i)(GridSearchCV|cross_val_score|cross_validate|validation_curve)','code'),
  ('evaluate','Classification performance',r'(?i)(accuracy_score|classification_report|f1_score|confusion_matrix|balanced_accuracy_score)','code'),
  ('writeup','Model explanation',r'(?i)(neighbor|neighbour|amenit|baseline|majority.class)','markdown'),
 ],
 'tree':[
  ('target','Host response time target',r'host_response_time','code'),
  ('model','Decision tree classifier',r'DecisionTreeClassifier\s*\(','code'),
  ('cv','Cross-validation / pruning',r'(?i)(GridSearchCV|cross_val_score|cross_validate|ccp_alpha|cost_complexity_pruning_path)','code'),
  ('display','Tree displayed',r'(plot_tree|export_graphviz|export_text)\s*\(','code'),
  ('metrics','Classification evaluation',r'(?i)(accuracy_score|classification_report|f1_score|confusion_matrix|balanced_accuracy_score)','code'),
  ('writeup','Tree interpretation',r'(?i)(tree|split|branch|response time|prun)','markdown'),
 ],
 'transformer':[
  ('quartile','Quartile target',r'(?i)(price_quartile|qcut\()','code'),
  ('texts','Three required text fields',r'(?s)(?=.*description)(?=.*host_about)(?=.*amenities)','code'),
  ('model','Transformer model',r'(?i)(transformers|AutoTokenizer|AutoModelForSequenceClassification|sentence_transformers|pipeline\()','code'),
  ('split','Train/test split',r'(?i)(train_test_split|train_dataset|test_dataset|DatasetDict)','code'),
  ('metrics','Held-out evaluation',r'(?i)(accuracy_score|classification_report|f1_score|compute_metrics|\.evaluate\()','code'),
  ('fiction','Fictional rental prediction',r'(?i)(fictional|hypothetical|example rental|made.up rental)','markdown'),
  ('writeup','Transformer interpretation',r'(?i)(transformer|quartile|validation|text classification)','markdown'),
 ],
 'clustering':[
  ('model','Clustering estimator',r'(?i)(KMeans\s*\(|AgglomerativeClustering\s*\(|linkage\s*\()','code'),
  ('features','Scaling / feature engineering',r'(?i)(StandardScaler|RobustScaler|MinMaxScaler|OneHotEncoder|feature.engineer)','code'),
  ('selection','Cluster-number assessment',r'(?i)(silhouette_score|davies_bouldin_score|inertia_|dendrogram)','code'),
  ('visuals','Cluster visualizations (three required; review)',r'(?i)(scatter|boxplot|barplot|violin|hist|countplot|pie)','code'),
  ('names','Cluster names and meanings',r'(?i)(cluster|segment|group)','markdown'),
 ],
 'conclusion':[
  ('big_picture','Overall insights and beneficiaries',r'(?i)(insight|stakeholder|benefit|host|guest|decision|limitation)','markdown'),
  ('submission','Submission readiness (human review)',r'(?i)(pdf|blackboard|script|submission)','markdown'),
 ],
}

def source(cell):
 s=cell.get('source','')
 return ''.join(s) if isinstance(s,list) else str(s)

def section_for(text):
 match=re.search(r'<!--\s*AD699:SECTION:([a-z]+)\s*-->',text,re.I)
 return match.group(1).lower() if match else None

def inspect_notebook(path):
 with path.open(encoding='utf-8') as f: nb=json.load(f)
 if nb.get('nbformat') != 4: raise ValueError('Expected a Jupyter .ipynb notebook (format 4).')
 data={k:{'code':[],'markdown':[],'outputs':0} for k,_ in SECTIONS}
 found=set(); active=None; cells=nb.get('cells',[])
 for idx,cell in enumerate(cells,1):
  val=source(cell)
  marker=section_for(val)
  if marker:
   active=marker if marker in data else None
   if active:found.add(active)
  if active and cell.get('cell_type') in ('code','markdown'):
   # Section headers and TODO placeholders are not completed work.
   if marker or re.search(r'(?im)^\s*(#\s*)?TODO\s*:',val):
    continue
   data[active][cell['cell_type']].append((idx,val))
   if cell.get('cell_type')=='code': data[active]['outputs']+=len(cell.get('outputs',[]))
 return nb,data,found

def execute_notebook(nb,path,cell_timeout):
 try:
  import nbformat
  from nbclient import NotebookClient
 except ImportError:
  return {'status':'not_run','reason':'Install nbformat, nbclient, and ipykernel to execute the notebook.'},None
 try:
  doc=nbformat.read(str(path),as_version=4)
  client=NotebookClient(doc,timeout=cell_timeout, kernel_name='python3',allow_errors=True,
                        resources={'metadata':{'path':str(path.parent.resolve())}})
  client.execute()
  errors=[]; outputs={}
  for i,c in enumerate(doc.cells,1):
   if c.cell_type=='code':
    outs=c.get('outputs',[])
    outputs[i]=len(outs)
    for o in outs:
     if o.get('output_type')=='error':errors.append({'cell':i,'type':o.get('ename','Error'),'message':o.get('evalue','')})
  return {'status':'passed' if not errors else 'failed','error_count':len(errors),'errors':errors[:30], 'executed_code_cells':len(outputs)},outputs
 except Exception as e:
  return {'status':'failed','error_count':1,'errors':[{'cell':None,'type':type(e).__name__,'message':str(e)}]},None

def evaluate(data,found,review,execution,outputs):
 sections=[]; all_checks=[]
 for key,label in SECTIONS:
  s=data[key]; checks=[]
  for cid,title,pat,ctype in RULES[key]:
   group=s[ctype]; matched=[idx for idx,text in group if re.search(pat,text)]
   detected=bool(matched)
   approved=bool(review.get(key,{}).get(cid,False))
   status='reviewed' if detected and approved else ('detected' if detected else 'missing')
   # Evaluation policy: detected = partial evidence, human approval = verified.
   checks.append({'id':cid,'title':title,'status':status,'detected':detected,'approved':approved,'cells':matched[:10]})
  detected_count=sum(x['detected'] for x in checks)
  approved_count=sum(x['status']=='reviewed' for x in checks)
  # 50% for evidence, 30% for approval, 20% for a clean execution (only when applicable).
  n=len(checks)
  execution_share=(0.2 if execution['status']=='passed' and key!='conclusion' and detected_count>0 else 0)
  progress=round(100*(.5*detected_count/n+.3*approved_count/n+execution_share))
  if key=='conclusion': progress=round(100*(.625*detected_count/n+.375*approved_count/n))
  if key not in found:progress=0
  if progress==100:status='Completed'
  elif progress>=90:status='Ready for review'
  elif progress>=50:status='Partially complete'
  elif progress>0:status='Started'
  else:status='Not started'
  sections.append({'id':key,'title':label,'progress':progress,'status':status,'checks':checks,
     'code_cells':len(s['code']),'markdown_cells':len(s['markdown']),'saved_outputs':s['outputs']})
  all_checks.extend(checks)
 return sections

def render(report):
 # Embed JSON directly so dashboard.html works even when opened as a local file.
 data=json.dumps(report,ensure_ascii=False).replace('</','<\\/')
 template=Path(__file__).with_name('dashboard_template.html').read_text(encoding='utf-8')
 return template.replace('/*__REPORT_JSON__*/null',data)

def main():
 parser=argparse.ArgumentParser(description='Validate AD699 Team C notebook and generate HTML dashboard')
 parser.add_argument('--notebook',default='AD699_Team_C_Semester_Project.ipynb')
 parser.add_argument('--output',default='Team_C_Progress_Dashboard.html')
 parser.add_argument('--review',default='review.json')
 parser.add_argument('--execute',action='store_true',help='Execute trusted notebook cells; may take a long time')
 parser.add_argument('--cell-timeout',type=int,default=180)
 args=parser.parse_args()
 path=Path(args.notebook).expanduser().resolve()
 if not path.is_file():parser.error(f'Notebook not found: {path}')
 nb,data,found=inspect_notebook(path)
 review_path=Path(args.review)
 review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
 execution={'status':'not_run','reason':'Use --execute to test that Python cells run.'}
 outputs=None
 execution_report = Path(
    "notebook_test/output/execution_report.json"
 )
 if execution_report.is_file():
    execution=json.loads(execution_report.read_text(encoding='utf-8'))
 if args.execute:
    execution,outputs=execute_notebook(nb,path,args.cell_timeout)
 sections=evaluate(data,found,review,execution,outputs)
 report={'project':'AD699 — Team C','notebook':path.name,'checked_at':dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
         'execution':execution,'sections':sections,'overall':round(sum(s['progress'] for s in sections)/len(sections)),
         'method':'Heuristic code/text detection + manual review + optional notebook execution. Not a guarantee of correctness.',
         'found_sections':sorted(found)}
 out=Path(args.output).expanduser().resolve()
 out.write_text(render(report),encoding='utf-8')
 out.with_name('Team_C_Progress_Report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('Dashboard:',out)
 print('Overall:',report['overall'],'% | Execution:',execution['status'])
 for s in sections:print(f" - {s['title']}: {s['progress']}% ({s['status']})")
 if execution.get('errors'):
  for e in execution['errors'][:5]:print('ERROR',e)
 return 0 if execution['status']!='failed' else 2
if __name__=='__main__':sys.exit(main())
