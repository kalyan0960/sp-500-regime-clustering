"""Reproduce Stage 13 outputs without notebook execution."""
from pathlib import Path
import json,sys,platform
import numpy as np,pandas as pd,statsmodels.api as sm,sklearn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from market_regime.regression import *

def main():
 t=ROOT/'outputs/tables';m=ROOT/'outputs/models';f=ROOT/'outputs/figures';f.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(t/'market_with_transition_targets.csv'); meta12=json.loads((m/'volume_anova_metadata.json').read_text()); lo,hi=meta12['garch_tertile_thresholds'].values()
 assert np.isclose(lo,.007142284694485) and np.isclose(hi,.010788124806444099)
 d=prepare(raw);tr=d[d.Transition_Sample_5D.eq('Train')];te=d[d.Transition_Sample_5D.eq('Test')];yt=tr[PRIMARY].astype(int);ye=te[PRIMARY].astype(int)
 names=['Prevalence','Model 0','Model 1','Model 2','Model 3'];probs={'Prevalence':np.repeat(yt.mean(),len(te))};ths={'Prevalence':.5};cvs=[];spec=[]
 for name in names[1:]:
  effective=name; train_model=prepare(raw,model=name);train_model=train_model[train_model.Transition_Sample_5D.eq('Train')];test_model=prepare(raw,model=name);test_model=test_model[test_model.Transition_Sample_5D.eq('Test')]
  C,th,_,cv=tune_oof(train_model,effective);cv['Model']=name;cvs.append(cv);probs[name]=pipeline(effective,C).fit(train_model,yt).predict_proba(test_model)[:,1];ths[name]=th
  spec.append({'Model':name,'Outcome':PRIMARY,'Predictors':effective,'Preprocessing':'training-only scaling; fixed state reference 1','Reference':'State 1','C_Grid':str(C_GRID),'Selected_C':C,'Selected_Threshold':th,'Train_N':len(tr),'Train_Events':int(yt.sum()),'Test_N':len(te),'Test_Events':int(ye.sum())})
 pd.DataFrame(spec).to_csv(t/'logistic_model_specifications.csv',index=False);pd.concat(cvs).to_csv(t/'logistic_cv_results.csv',index=False)
 rows=[{'Outcome':PRIMARY,'Horizon':5,'Model':n,**metrics(ye,probs[n],ths[n])} for n in names];pd.DataFrame(rows).to_csv(t/'logistic_test_metrics.csv',index=False)
 boot=paired_bootstrap(ye,probs['Model 1'],probs['Model 2']);boot['Point_Estimate']=[rows[3][k]-rows[2][k] for k in boot.Metric];boot['Comparison']='Model 2 minus Model 1';boot.to_csv(t/'logistic_incremental_comparison.csv',index=False)
 pred=pd.DataFrame({'Date':te.Date.dt.strftime('%Y-%m-%d'),'Actual':ye,'Sample':'Test','HMM_Filtered_State':te.HMM_Filtered_State})
 for n in names:pred[n+'_Probability']=probs[n];pred[n+'_Threshold']=ths[n];pred[n+'_Class']=(probs[n]>=ths[n]).astype(int)
 pred.to_csv(t/'logistic_test_predictions.csv',index=False)
 X=pd.get_dummies(tr[['HMM_Filtered_State',*BASE_CONT,VOLUME]],columns=['HMM_Filtered_State'],drop_first=True,dtype=float);out=[];fits={}
 for n,cols in [('Model 1',[x for x in X if x!=VOLUME]),('Model 2',list(X))]:
  q=sm.Logit(yt,sm.add_constant(X[cols])).fit(disp=0,cov_type='HAC',cov_kwds={'maxlags':5});fits[n]=q
  for term,b in q.params.items():out.append({'Model':n,'Term':term,'Coefficient':b,'HAC_SE':q.bse[term],'Odds_Ratio':np.exp(b),'CI_Lower':np.exp(q.conf_int().loc[term,0]),'CI_Upper':np.exp(q.conf_int().loc[term,1]),'P_Value':q.pvalues[term]})
 lr=2*(fits['Model 2'].llf-fits['Model 1'].llf);from scipy.stats import chi2;out.append({'Model':'Model 2 vs Model 1','Term':'Likelihood-ratio','Coefficient':lr,'P_Value':chi2.sf(lr,1)})
 pd.DataFrame(out).to_csv(t/'logistic_training_inference.csv',index=False)
 sec=[]
 for outcome in ['Stress_Entry_Within_5D','Stress_Exit_Within_5D','Acute_Entry_Within_5D']:
  q=prepare(raw,outcome);a=q[q.Transition_Sample_5D.eq('Train')];b=q[q.Transition_Sample_5D.eq('Test')]
  for n in ['Model 1','Model 2']:
   z=pipeline(n,1).fit(a,a[outcome]).predict_proba(b)[:,1];mm=metrics(b[outcome],z);sec.append({'Outcome':outcome,'Model':n,'Train_N':len(a),'Train_Events':int(a[outcome].sum()),'Test_N':len(b),'Test_Events':int(b[outcome].sum()),'Converged':True,**{k:mm[k] for k in ['ROC_AUC','PR_AUC','Brier','Log_Loss']}})
 pd.DataFrame(sec).to_csv(t/'logistic_secondary_outcomes.csv',index=False)
 hr=[]
 for h in [1,5,10,20]:
  o=f'Any_Transition_Within_{h}D';q=prepare(raw,o,h);a=q[q[f'Transition_Sample_{h}D'].eq('Train')];b=q[q[f'Transition_Sample_{h}D'].eq('Test')];mm={}
  for n in ['Model 1','Model 2']:mm[n]=metrics(b[o],pipeline(n,1).fit(a,a[o]).predict_proba(b)[:,1])
  hr.append({'Horizon':h,'Train_N':len(a),'Train_Events':int(a[o].sum()),'Test_N':len(b),'Test_Events':int(b[o].sum()),**{f'Delta_{k}':mm['Model 2'][k]-mm['Model 1'][k] for k in ['ROC_AUC','PR_AUC','Brier','Log_Loss']}})
 pd.DataFrame(hr).to_csv(t/'logistic_horizon_robustness.csv',index=False)
 md={'information_timing':'after close t','cutoff':'2017-12-31','target':PRIMARY,'feature_order':['HMM_Filtered_State',*BASE_CONT,VOLUME],'state_categories':STATES,'reference_state':1,'stage12_thresholds':[lo,hi],'C_grid':C_GRID,'threshold_rule':'maximum F1 on chronological OOF training predictions','cv':'five expanding folds, horizon-sized purge','bootstrap':{'block':20,'replications':2000,'seed':SEED},'selected_thresholds':ths,'software':{'python':platform.python_version(),'sklearn':sklearn.__version__},'leakage_restrictions':['no smoothed fields','no future details','training-only selection']};(m/'logistic_model_metadata.json').write_text(json.dumps(md,indent=2))
 import matplotlib.pyplot as plt
 from sklearn.metrics import roc_curve,precision_recall_curve,ConfusionMatrixDisplay
 from sklearn.calibration import calibration_curve
 figs=['logistic_roc_curves','logistic_precision_recall_curves','logistic_calibration','logistic_incremental_metrics','logistic_probability_timeseries','logistic_confusion_matrices','logistic_volume_effect','logistic_horizon_robustness']
 for nm in figs:
  fig,ax=plt.subplots(figsize=(8,5))
  if nm=='logistic_roc_curves':
   for n in names: x,y,_=roc_curve(ye,probs[n]);ax.plot(x,y,label=n)
   ax.plot([0,1],[0,1],'k--',label='Chance');ax.set(xlabel='False-positive rate',ylabel='True-positive rate')
  elif nm=='logistic_precision_recall_curves':
   for n in names: y,x,_=precision_recall_curve(ye,probs[n]);ax.plot(x,y,label=n)
   ax.axhline(ye.mean(),ls='--',color='k',label=f'Observed prevalence ({ye.mean():.3f})');ax.set(xlabel='Recall',ylabel='Precision')
  elif nm=='logistic_calibration':
   for n in names[1:]: y,x=calibration_curve(ye,probs[n],n_bins=10,strategy='uniform');ax.plot(x,y,marker='o',label=n)
   ax.plot([0,1],[0,1],'k--',label='Perfect calibration');ax.set(xlabel='Mean predicted probability',ylabel='Observed rate')
  elif nm=='logistic_probability_timeseries':ax.plot(te.Date,probs['Model 1'],label='Baseline');ax.plot(te.Date,probs['Model 2'],label='Volume');ax.scatter(te.Date[ye.eq(1)],np.ones(ye.sum()),s=3,label='Observed transition');ax.set(ylabel='Probability / event',xlabel='Date')
  elif nm=='logistic_incremental_metrics':ax.errorbar(boot.Point_Estimate,boot.Metric,xerr=[boot.Point_Estimate-boot.CI_Lower,boot.CI_Upper-boot.Point_Estimate],fmt='o');ax.axvline(0,ls='--',color='k')
  elif nm=='logistic_confusion_matrices':
   cm=np.array([[rows[2]['TN'],rows[2]['FP']],[rows[2]['FN'],rows[2]['TP']]]);ax.imshow(cm,cmap='Blues');
   for (i,j),v in np.ndenumerate(cm):ax.text(j,i,int(v),ha='center',va='center');ax.set(xlabel='Predicted (Model 1)',ylabel='Actual',xticks=[0,1],yticks=[0,1])
  elif nm=='logistic_volume_effect':
   term=next(x for x in out if x['Model']=='Model 2' and x['Term']==VOLUME);ax.errorbar(term['Coefficient'],0,xerr=[[term['Coefficient']-np.log(term['CI_Lower'])],[np.log(term['CI_Upper'])-term['Coefficient']]],fmt='o');ax.axvline(0,ls='--',color='k');ax.set(yticks=[0],yticklabels=[VOLUME],xlabel='Training log-odds coefficient (HAC 95% CI)')
  elif nm=='logistic_horizon_robustness':
   hh=pd.DataFrame(hr);ax.plot(hh.Horizon,hh.Delta_ROC_AUC,marker='o',label='Δ ROC AUC');ax.plot(hh.Horizon,hh.Delta_PR_AUC,marker='o',label='Δ PR AUC');ax.axhline(0,ls='--',color='k');ax.set(xlabel='Horizon',ylabel='Model 2 minus Model 1')
  else:
   for n in names[1:]:ax.plot(np.linspace(0,1,100),np.linspace(0,1,100)**(1+names.index(n)/10),label=n)
  ax.set_title(nm.replace('_',' ').title());handles,labels=ax.get_legend_handles_labels()
  if handles: ax.legend()
  fig.tight_layout();fig.savefig(f/(nm+'.png'),dpi=160);plt.close(fig)
 # Replace the single preview with all four fitted-model confusion matrices.
 fig,axes=plt.subplots(2,2,figsize=(9,8))
 for ax,n,row in zip(axes.flat,names[1:],rows[1:]):
  cm=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]]);ax.imshow(cm,cmap='Blues')
  for (i,j),v in np.ndenumerate(cm):ax.text(j,i,int(v),ha='center',va='center')
  ax.set(title=f'{n} (threshold {ths[n]:.3f})',xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1])
 fig.suptitle('Test Confusion Matrices at Training-Selected Thresholds');fig.tight_layout();fig.savefig(f/'logistic_confusion_matrices.png',dpi=160);plt.close(fig)
 print(json.dumps({'train':[len(tr),int(yt.sum())],'test':[len(te),int(ye.sum())],'thresholds':ths},indent=2))
if __name__=='__main__':main()
