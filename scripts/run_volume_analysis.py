"""Run deterministic Stage 12 training-only abnormal-volume analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests

from market_regime.volume_analysis import *
from market_regime.plots import create_volume_analysis_figures

ROOT=Path(__file__).resolve().parents[1]; TABLES=ROOT/"outputs"/"tables"; MODELS=ROOT/"outputs"/"models"

def main():
    frame=pd.read_csv(TABLES/"market_with_transition_targets.csv"); sample,thresholds=prepare_primary(frame)
    summaries=[group_summary(sample,PRIMARY_TARGET,OUTCOME,"Primary training",True),
               group_summary(sample,PRIMARY_TARGET,"Abnormal_Volume","Raw-volume robustness",False)]
    high=sample.loc[sample.Transition_Window_All_High_Confidence_5D.eq(True)]
    summaries.append(group_summary(high,PRIMARY_TARGET,OUTCOME,"All-high-confidence window",False))
    robust=[primary_effects(sample),primary_effects(sample,outcome="Abnormal_Volume",restriction="Raw-volume robustness"),
            primary_effects(high,restriction="All-high-confidence window")]
    for target in SECONDARY_TARGETS:
        summaries.append(group_summary(sample,target,OUTCOME,"Secondary training",False))
        robust.append(primary_effects(sample,target=target,restriction="Secondary training"))
    anova,hac,model=fit_two_factor(sample)
    if anova.loc[(anova.Inference_Method=="Classical Type III ANOVA") & anova.Term.ne("Residual"),"P_Value"].min()<.05:
        gh=pg.pairwise_gameshowell(data=sample.assign(Cell=sample.Transition.astype(str)+"_"+sample.Volatility_Category.astype(str)),
            dv=OUTCOME,between="Cell").rename(columns={"T":"Statistic","pval":"P_Value","hedges":"Hedges_g"})
        gh["Adjusted_P_Value"]=multipletests(gh.P_Value,method="holm")[1]; gh["Test"]="Games-Howell (Holm adjusted)"
        gh["Analysis"]="Six-cell post-hoc"; gh["Target"]=PRIMARY_TARGET; gh["Outcome"]=OUTCOME; gh["Sample_Restriction"]="Primary training"
        robust.append(gh)
    bootstrap=block_bootstrap(sample,reps=2000)
    windows=event_windows(frame); event=event_summary(windows)
    group=pd.concat(summaries,ignore_index=True); tests=pd.concat(robust,ignore_index=True,sort=False)
    confidence_rows=[]
    for restriction,data in (("Full target sample",sample),("All-high-confidence window",high)):
        effect=primary_effects(data).iloc[0]
        confidence_rows.append({"Restriction":restriction,"N":len(data),"Transition_Count":int(data.Transition.sum()),
            "Mean_Difference":effect.Mean_Difference,"Median_Difference":effect.Median_Difference,
            "Hedges_g":effect.Hedges_g,"Rank_Effect":effect.Rank_Effect})
    pd.DataFrame(confidence_rows).to_csv(TABLES/"volume_confidence_robustness.csv",index=False)
    group.to_csv(TABLES/"volume_transition_group_summary.csv",index=False); anova.to_csv(TABLES/"volume_two_way_anova.csv",index=False)
    hac.to_csv(TABLES/"volume_hac_regression.csv",index=False); tests.to_csv(TABLES/"volume_robust_tests.csv",index=False)
    bootstrap.to_csv(TABLES/"volume_block_bootstrap.csv",index=False); windows.to_csv(TABLES/"transition_event_windows.csv",index=False)
    event.to_csv(TABLES/"transition_event_summary.csv",index=False)
    metadata={"training_cutoff":"2017-12-31","primary_outcome":OUTCOME,"primary_target":PRIMARY_TARGET,
        "garch_tertile_thresholds":{"low_medium":thresholds[0],"medium_high":thresholds[1]},"anova_type":"Type III",
        "contrasts":"Treatment contrasts; Transition=0 and Volatility=Low references","hac_max_lag":5,
        "block_length":20,"bootstrap_seed":12012,"bootstrap_replications":2000,"multiple_testing_adjustment":"Holm",
        "confidence_level":.95,"primary_sample_size":len(sample),"transition_count":int(sample.Transition.sum()),
        "limitations":["Association, not causation or out-of-sample prediction","Overlapping targets induce dependence",
                       "HC3 is not serial-correlation robust","Confidence filtering can remove genuine boundary events"]}
    (MODELS/"volume_anova_metadata.json").write_text(json.dumps(metadata,indent=2))
    create_volume_analysis_figures(sample,group,bootstrap,event,tests,pd.DataFrame(confidence_rows),ROOT/"outputs"/"figures")
    print(json.dumps(metadata,indent=2))

if __name__=="__main__": main()
