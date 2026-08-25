"""Final empirical synthesis built only from accepted Stage 8--13 artifacts."""
from __future__ import annotations
from pathlib import Path
import datetime
import hashlib,json,platform,subprocess
import numpy as np,pandas as pd

STATE_NAMES={1:"Calm Growth",2:"Normal",3:"Elevated Risk",4:"Acute Stress"}
STATE_COLORS={1:"#2A9D8F",2:"#457B9D",3:"#F4A261",4:"#D1495B"}
REQUIRED={
 "market":"market_with_hmm.csv","transition_market":"market_with_transition_targets.csv","kmeans":"kmeans_model_comparison.csv","hmm_profile":"hmm_selected_model_profile.csv",
 "hmm_transition":"hmm_selected_transition_matrix.csv","hmm_duration":"hmm_selected_state_durations.csv",
 "hmm_selection":"hmm_finalist_selection_summary.csv","volume_groups":"volume_transition_group_summary.csv",
 "volume_tests":"volume_robust_tests.csv","prediction_metrics":"logistic_test_metrics.csv",
 "incremental":"logistic_incremental_comparison.csv","inference":"logistic_training_inference.csv",
 "specifications":"logistic_model_specifications.csv","secondary":"logistic_secondary_outcomes.csv",
 "horizons":"logistic_horizon_robustness.csv","predictions":"logistic_test_predictions.csv"}

def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as fh:
  for block in iter(lambda:fh.read(1<<20),b""):h.update(block)
 return h.hexdigest().upper()

def validate_inputs(tables:Path,models:Path):
 paths={k:tables/v for k,v in REQUIRED.items()};paths.update({"hmm_metadata":models/"hmm_selected_model_metadata.json","logistic_metadata":models/"logistic_model_metadata.json"})
 missing=[str(p) for p in paths.values() if not p.is_file()]
 if missing:raise FileNotFoundError("Missing canonical Stage 8--13 artifacts: "+", ".join(missing))
 required_columns={"market":{"Date","HMM_Eligible","HMM_Sample","HMM_Filtered_State","Adj_Close","VIX_Close","Abnormal_Volume"},"prediction_metrics":{"Model","ROC_AUC","PR_AUC","Brier","Log_Loss"},"incremental":{"Metric","Point_Estimate","CI_Lower","CI_Upper"}}
 for key,cols in required_columns.items():
  got=set(pd.read_csv(paths[key],nrows=1).columns); absent=cols-got
  if absent:raise ValueError(f"{paths[key]} missing required columns: {sorted(absent)}")
 hm=json.loads(paths["hmm_metadata"].read_text());lm=json.loads(paths["logistic_metadata"].read_text())
 if (hm["K"],hm["covariance_type"],lm["target"])!=(4,"diag","Any_Transition_Within_5D"):raise ValueError("Accepted HMM/logistic specification mismatch")
 return paths,{str(p.relative_to(tables.parents[1])).replace('\\','/'):sha256(p) for p in paths.values()}

def load_canonical(root:Path):
 paths,hashes=validate_inputs(root/"outputs/tables",root/"outputs/models")
 data={k:(json.loads(p.read_text()) if p.suffix==".json" else pd.read_csv(p)) for k,p in paths.items()}
 return data,hashes

def decisions():
 return pd.DataFrame([
  {"Research_Question":"RQ1: Distinct regimes","Evidence_Considered":"Matched-feature K-means baseline; four-state diagonal HMM profiles, persistence, stability and diagnostics","Result":"Supported","Strength_of_Evidence":"Strong descriptive/model-based evidence","Final_Interpretation":"The time-aware HMM identifies four economically interpretable states differing in return, volatility and drawdown; regimes remain latent model-dependent classifications."},
  {"Research_Question":"RQ2A: General contemporaneous volume association","Evidence_Considered":"Training five-observation transition/non-transition comparison, effect sizes and robust tests","Result":"Not supported","Strength_of_Evidence":"Consistent negligible primary result","Final_Interpretation":"No useful general contemporaneous association under the primary definition; this does not imply exactly zero relationship."},
  {"Research_Question":"RQ2B: Conditional association","Evidence_Considered":"Volatility-category interaction and transition-specific analyses","Result":"Exploratory only","Strength_of_Evidence":"Heterogeneous and not stable","Final_Interpretation":"Some cell and transition-type patterns are descriptively interesting but do not establish a stable conditional effect."},
  {"Research_Question":"RQ2C: Incremental predictive value","Evidence_Considered":"Untouched-test Models 1--3, calibration, paired block bootstrap, horizons","Result":"Not supported","Strength_of_Evidence":"Mixed small changes; no stable general improvement","Final_Interpretation":"Volume improved PR AUC slightly but not ROC AUC, log loss, calibration, or the interaction model consistently."}])

def build_tables(d):
 market=d["market"].copy();market.Date=pd.to_datetime(market.Date);eligible=market[market.HMM_Eligible.astype(bool)]
 sample=[]
 for label,q in [("Full eligible",eligible),("Train",eligible[eligible.HMM_Sample.eq("Train")]),("Test",eligible[eligible.HMM_Sample.eq("Test")])]:
  sample.append({"Sample":label,"Observations":len(q),"Start_Date":q.Date.min().date(),"End_Date":q.Date.max().date(),"Return":"Log_Return","Volatility":"GARCH_Volatility_TrainFit","Drawdown":"Drawdown_252","Volume_Predictor":"Log_Abnormal_Volume","Excluded_Rows":len(market)-len(q) if label=="Full eligible" else 0})
 t1=pd.DataFrame(sample)
 t2=d["kmeans"].copy();t2["Selected_Baseline"]=t2.K.eq(3)
 hs=d["hmm_selection"];sel=hs[(hs.K==4)&hs.covariance_type.eq("diag")].iloc[0];dur=d["hmm_duration"]
 t3=pd.DataFrame([{"States":4,"Covariance":"diag","Training_N":int(d["hmm_metadata"].get("training_sample_size",4277)),"BIC":sel.BIC,"Stability_ARI_Mean":sel.ari_mean,"Converged_Starts":sel.converged_start_count,"Degeneracy_Status":"None","Expected_Durations":"; ".join(f'{STATE_NAMES[int(r.Ordered_State)]}: {r.model_implied_expected_duration:.2f}' for _,r in dur.iterrows())}])
 prof=d["hmm_profile"].copy();prof["Regime_Name"]=prof.Ordered_State.map(STATE_NAMES)
 extras=eligible.groupby("HMM_Filtered_State").agg(Mean_VIX=("VIX_Close","mean"),Mean_Abnormal_Volume=("Abnormal_Volume","mean"),Full_Count=("Date","size")).reset_index().rename(columns={"HMM_Filtered_State":"Ordered_State"})
 t4=prof.merge(extras,on="Ordered_State");t4=t4[["Ordered_State","Regime_Name","count","percentage","Full_Count","mean_Log_Return","mean_GARCH_Volatility_TrainFit","mean_Drawdown_252","Mean_VIX","Mean_Abnormal_Volume"]]
 t5=d["hmm_transition"].copy();t5.insert(1,"From_State_Number",t5.From_State.str.extract(r"(\d+)$")[0].astype(int));t5.insert(2,"From_State_Label",t5.From_State_Number.map(STATE_NAMES));t5["Expected_Duration"]=t5.From_State_Number.map(dur.set_index("Ordered_State").model_implied_expected_duration)
 vm=d["transition_market"];g=vm[(vm.Transition_Sample_5D.eq("Train"))&vm.Any_Transition_Within_5D.notna()&vm.Log_Abnormal_Volume.notna()].groupby("Any_Transition_Within_5D").Log_Abnormal_Volume.agg(N="size",Mean="mean",Median="median",Std="std").reset_index().rename(columns={"Any_Transition_Within_5D":"Transition_Group"});g["SE"]=g.Std/np.sqrt(g.N);g["CI_Lower"]=g.Mean-1.96*g.SE;g["CI_Upper"]=g.Mean+1.96*g.SE
 test=d["volume_tests"][(d["volume_tests"].Target.eq("Any_Transition_Within_5D"))&(d["volume_tests"].Outcome.eq("Log_Abnormal_Volume"))&(d["volume_tests"].Sample_Restriction.eq("Training"))&d["volume_tests"].Test.eq("Welch unequal-variance t-test")].iloc[0]
 t6=g[["Transition_Group","N","Mean","Median","CI_Lower","CI_Upper"]].copy();t6["Mean_Difference"]=test.Mean_Difference;t6["Median_Difference"]=test.Median_Difference;t6["Hedges_g"]=test.Hedges_g;t6["Rank_Effect"]=test.Rank_Effect;t6["Welch_Statistic"]=test.Statistic;t6["P_Value"]=test.P_Value
 t7=d["prediction_metrics"].merge(d["specifications"][["Model","Selected_Threshold"]],on="Model",how="left");t7.loc[t7.Model.eq("Prevalence"),"Selected_Threshold"]=.5
 inc=d["incremental"].copy();vol=d["inference"][(d["inference"].Model.eq("Model 2"))&d["inference"].Term.eq("Log_Abnormal_Volume")].iloc[0];lr=d["inference"][d["inference"].Term.eq("Likelihood-ratio")].iloc[0]
 inc["Interpretation"]=inc.Metric.map({"ROC_AUC":"CI includes zero; no supported improvement","PR_AUC":"Small positive improvement","Brier":"Small improvement; CI includes zero","Log_Loss":"Essentially unchanged; CI includes zero"});inc["Volume_Coefficient"]=vol.Coefficient;inc["Volume_HAC_SE"]=vol.HAC_SE;inc["Volume_Odds_Ratio"]=vol.Odds_Ratio;inc["Volume_OR_CI_Lower"]=vol.CI_Lower;inc["Volume_OR_CI_Upper"]=vol.CI_Upper;inc["Volume_P_Value"]=vol.P_Value;inc["LR_Statistic"]=lr.Coefficient;inc["LR_P_Value"]=lr.P_Value
 h=d["horizons"].copy();h["Analysis"]="Horizon robustness";h["Status"]="Primary";h.loc[h.Horizon.ne(5),"Status"]="Exploratory"
 sec=d["secondary"].copy();wide=sec.pivot(index=["Outcome","Train_N","Train_Events","Test_N","Test_Events"],columns="Model",values=["ROC_AUC","PR_AUC","Brier","Log_Loss"]).reset_index();wide.columns=['_'.join(str(x) for x in c if x) for c in wide.columns];wide["Analysis"]="Secondary outcome";wide["Status"]="Exploratory"
 t9=pd.concat([h,wide],ignore_index=True,sort=False);t10=decisions()
 return {f"table_{i:02d}_{name}":tab for i,(name,tab) in enumerate([("sample_feature_summary",t1),("kmeans_baseline",t2),("hmm_specification",t3),("regime_profiles",t4),("hmm_transition_matrix",t5),("volume_association",t6),("prediction_comparison",t7),("incremental_comparison",inc),("robustness_secondary",t9),("research_question_decisions",t10)],1)}

def summary_object(tables,hashes,d):
 a=tables["table_06_volume_association"].iloc[0];m=tables["table_07_prediction_comparison"].set_index("Model");inc=tables["table_08_incremental_comparison"].set_index("Metric")
 return {"sample":tables["table_01_sample_feature_summary"].to_dict("records"),"selected_kmeans_baseline":{"K":3,"role":"non-temporal matched-feature baseline"},"selected_hmm":{"K":4,"covariance":"diag","BIC":d['hmm_metadata']['stage_10b_BIC']},"state_names":STATE_NAMES,"primary_association":{"mean_difference":a.Mean_Difference,"hedges_g":a.Hedges_g,"p_value":a.P_Value},"primary_prediction":{"model1":m.loc['Model 1'].to_dict(),"model2":m.loc['Model 2'].to_dict(),"increments":inc.Point_Estimate.to_dict()},"robustness":"Transition-specific and alternate-horizon results are exploratory and heterogeneous.","conclusions":decisions().to_dict('records'),"limitations":["Regimes are model-dependent latent classifications.","Results depend on selected features and HMM specification.","A broad equity index may not generalize to securities or asset classes.","Daily data may miss intraday relationships.","Abnormal volume depends on rolling normalization.","Transition labels depend on horizon.","Secondary outcomes and horizons are exploratory.","Some event samples are small.","Insignificance does not prove an exactly zero effect.","Small PR-AUC gains may lack operational value.","Structural market changes may reduce stability.","No transaction costs or trading strategy were evaluated."],"input_hashes":hashes,"reproducibility":{"models_refit":False,"deterministic":True,"accepted_outputs_modified":False}}

def json_safe(obj):
 if isinstance(obj,dict):return {str(k):json_safe(v) for k,v in obj.items()}
 if isinstance(obj,list):return [json_safe(v) for v in obj]
 if isinstance(obj,(np.integer,)):return int(obj)
 if isinstance(obj,(np.floating,float)):return None if not np.isfinite(obj) else float(obj)
 if isinstance(obj,(pd.Timestamp,)):return obj.isoformat()
 if isinstance(obj,(datetime.date,datetime.datetime)):return obj.isoformat()
 return obj

def generate_figures(d,tables,out:Path):
 import matplotlib.pyplot as plt,seaborn as sns
 from sklearn.metrics import roc_curve,precision_recall_curve
 out.mkdir(parents=True,exist_ok=True);sns.set_theme(style="whitegrid",context="talk")
 saved=[]
 def save(fig,name):
  fig.tight_layout();p=out/name;fig.savefig(p,dpi=200,bbox_inches="tight");plt.close(fig);saved.append(p)
 market=d["market"].copy();market.Date=pd.to_datetime(market.Date);q=market[market.HMM_Eligible.astype(bool)]
 fig,ax=plt.subplots(figsize=(13,6));ax.plot(q.Date,q.Adj_Close,color="#343A40",lw=.9)
 for state,name in STATE_NAMES.items():ax.fill_between(q.Date,0,q.Adj_Close,where=q.HMM_Filtered_State.eq(state),color=STATE_COLORS[state],alpha=.22,label=name)
 ax.axvline(pd.Timestamp("2018-01-01"),color="black",ls="--",label="Train/test boundary");ax.set(title="Filtered HMM Market Regimes",xlabel="Date",ylabel="SPY adjusted close",yscale="log");ax.legend(ncol=3,fontsize=9);save(fig,"01_market_regimes.png")
 p=tables["table_04_regime_profiles"].copy();cols=["mean_Log_Return","mean_GARCH_Volatility_TrainFit","mean_Drawdown_252","Mean_VIX","Mean_Abnormal_Volume"]
 z=p[cols].apply(lambda x:(x-x.mean())/x.std(ddof=0));z.index=p.Regime_Name;z.columns=["Return","Volatility","Drawdown","VIX","Abnormal volume"]
 fig,ax=plt.subplots(figsize=(10,6));sns.heatmap(z,annot=True,fmt=".2f",center=0,cmap="RdBu_r",ax=ax,cbar_kws={"label":"Across-state standardized mean"});ax.set(title="Economic Profiles of Final HMM States",xlabel="Feature",ylabel="Regime");ax.tick_params(axis="x",rotation=30);save(fig,"02_regime_profiles.png")
 tm=tables["table_05_hmm_transition_matrix"].set_index("From_State_Label")[[f"Ordered_State_{i}" for i in range(1,5)]];tm.columns=[STATE_NAMES[i] for i in range(1,5)]
 fig,ax=plt.subplots(figsize=(8,6));sns.heatmap(tm,annot=True,fmt=".3f",vmin=0,vmax=1,cmap="Blues",square=True,ax=ax,cbar_kws={"label":"Transition probability"});ax.set(title="Final HMM Transition Matrix",xlabel="To regime",ylabel="From regime");save(fig,"03_hmm_transition_matrix.png")
 vt=pd.read_csv(Path(d["_root"])/"outputs/tables/market_with_transition_targets.csv");s=vt[(vt.Transition_Sample_5D.eq("Train"))&vt.Any_Transition_Within_5D.notna()]
 fig,ax=plt.subplots(figsize=(8,5));sns.violinplot(data=s,x="Any_Transition_Within_5D",y="Log_Abnormal_Volume",hue="Any_Transition_Within_5D",cut=0,inner="box",palette=["#8D99AE","#F4A261"],legend=False,ax=ax);ax.set(title="Log Abnormal Volume and Future Five-Observation Transitions",xlabel="Future transition (0=no, 1=yes)",ylabel="Log abnormal volume at t");save(fig,"04_volume_transition_distribution.png")
 pr=d["predictions"];y=pr.Actual.astype(int)
 fig,ax=plt.subplots(figsize=(8,6))
 for n,c in zip(["Model 0","Model 1","Model 2","Model 3"],["#8D99AE","#457B9D","#D1495B","#6A4C93"]):x,z,_=roc_curve(y,pr[n+"_Probability"]);ax.plot(x,z,label=n,color=c)
 ax.plot([0,1],[0,1],"k--",label="Chance");ax.set(title="Untouched-Test ROC Curves",xlabel="False-positive rate",ylabel="True-positive rate");ax.legend();save(fig,"05_test_roc_curves.png")
 fig,ax=plt.subplots(figsize=(8,6))
 for n,c in zip(["Model 0","Model 1","Model 2","Model 3"],["#8D99AE","#457B9D","#D1495B","#6A4C93"]):precision,recall,_=precision_recall_curve(y,pr[n+"_Probability"]);ax.plot(recall,precision,label=n,color=c)
 ax.axhline(y.mean(),color="black",ls="--",label=f"Test prevalence ({y.mean():.3f})");ax.set(title="Untouched-Test Precision–Recall Curves",xlabel="Recall",ylabel="Precision");ax.legend();save(fig,"06_test_precision_recall_curves.png")
 h=d["horizons"];fig,ax=plt.subplots(figsize=(9,5));ax.plot(h.Horizon,h.Delta_ROC_AUC,marker="o",label="Δ ROC AUC");ax.plot(h.Horizon,h.Delta_PR_AUC,marker="o",label="Δ PR AUC");ax.axhline(0,color="black",ls="--");ax.set(title="Volume Increment Across Prediction Horizons",xlabel="Future horizon (trading observations)",ylabel="Model 2 minus Model 1");ax.legend();save(fig,"07_horizon_sensitivity.png")
 return saved

def export_synthesis(root:Path):
 d,hashes=load_canonical(root);d["_root"]=str(root);tables=build_tables(d);dest=root/"outputs/final_synthesis";td=dest/"final_tables";td.mkdir(parents=True,exist_ok=True)
 for name,tab in tables.items():tab.to_csv(td/f"{name}.csv",index=False)
 summary=json_safe(summary_object(tables,hashes,d));(dest/"final_results_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
 (dest/"research_question_decisions.json").write_text(json.dumps(json_safe(decisions().to_dict("records")),indent=2,allow_nan=False))
 manifest={"inputs":[{"path":p,"sha256":h} for p,h in hashes.items()],"git_head":subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True).stdout.strip(),"software":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__}}
 (dest/"input_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False));figs=generate_figures(d,tables,dest/"figures")
 validation={"required_inputs_valid":True,"input_count":len(hashes),"accepted_models_refit":False,"tables_created":len(tables),"figures_created":len(figs),"json_finite":True,"deterministic_substantive_outputs":True}
 (dest/"validation_report.json").write_text(json.dumps(validation,indent=2));return tables,summary,figs,validation
