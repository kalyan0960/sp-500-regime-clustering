"""Training-only abnormal-volume association analysis for Stage 12."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

PRIMARY_TARGET = "Any_Transition_Within_5D"
OUTCOME = "Log_Abnormal_Volume"
VOLATILITY = "GARCH_Volatility_TrainFit"
VOL_LABELS = ["Low", "Medium", "High"]
SECONDARY_TARGETS = ["Stress_Entry_Within_5D", "Stress_Exit_Within_5D",
                     "Acute_Entry_Within_5D", "Stress_Worsening_Within_5D",
                     "Stress_Improving_Within_5D"]
FORMULA = ("Log_Abnormal_Volume ~ C(Transition, Treatment(reference=0)) * "
           "C(Volatility_Category, Treatment(reference='Low'))")


def primary_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Return prespecified finite Stage 12 training observations only."""
    mask = (frame["Transition_Sample_5D"].eq("Train") & frame[PRIMARY_TARGET].notna()
            & np.isfinite(frame[OUTCOME]) & np.isfinite(frame[VOLATILITY]))
    return frame.loc[mask].copy()


def training_tertiles(training: pd.DataFrame) -> tuple[float, float]:
    values = training[VOLATILITY].to_numpy(float)
    return tuple(float(x) for x in np.quantile(values, [1/3, 2/3]))


def assign_volatility_category(values: pd.Series, thresholds: tuple[float, float]) -> pd.Categorical:
    return pd.cut(values, [-np.inf, thresholds[0], thresholds[1], np.inf], labels=VOL_LABELS,
                  include_lowest=True, ordered=True)


def prepare_primary(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    sample = primary_sample(frame); thresholds = training_tertiles(sample)
    sample["Transition"] = sample[PRIMARY_TARGET].astype(int)
    sample["Volatility_Category"] = assign_volatility_category(sample[VOLATILITY], thresholds)
    return sample, thresholds


def group_summary(frame: pd.DataFrame, target: str, outcome: str, restriction: str,
                  volatility_cells: bool = False) -> pd.DataFrame:
    applicable = frame.loc[frame[target].notna() & np.isfinite(frame[outcome])].copy()
    groups = [target] + (["Volatility_Category"] if volatility_cells else [])
    rows = []
    for keys, values in applicable.groupby(groups, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,); n = len(values); mean = values[outcome].mean()
        se = values[outcome].std(ddof=1) / np.sqrt(n); critical = stats.t.ppf(.975, n-1) if n > 1 else np.nan
        rows.append({"Target": target, "Sample_Restriction": restriction, "Outcome": outcome,
            "Transition_Group": int(keys[0]), "Volatility_Category": keys[1] if len(keys)>1 else "All",
            "N": n, "Mean": mean, "Median": values[outcome].median(), "Std": values[outcome].std(ddof=1),
            "SE": se, "CI_Lower": mean-critical*se, "CI_Upper": mean+critical*se})
    return pd.DataFrame(rows)


def hedges_g(x1, x0) -> float:
    x1, x0 = np.asarray(x1,float), np.asarray(x0,float); n1,n0=len(x1),len(x0)
    pooled=np.sqrt(((n1-1)*x1.var(ddof=1)+(n0-1)*x0.var(ddof=1))/(n1+n0-2))
    d=(x1.mean()-x0.mean())/pooled; return float((1-3/(4*(n1+n0)-9))*d)


def cliff_delta(x1, x0) -> float:
    u=stats.mannwhitneyu(x1,x0,alternative="two-sided").statistic
    return float(2*u/(len(x1)*len(x0))-1)


def primary_effects(sample: pd.DataFrame, target=PRIMARY_TARGET, outcome=OUTCOME,
                    restriction="Training") -> pd.DataFrame:
    data=sample.loc[sample[target].notna() & np.isfinite(sample[outcome])]
    x1=data.loc[data[target].eq(1),outcome].to_numpy(); x0=data.loc[data[target].eq(0),outcome].to_numpy()
    welch=stats.ttest_ind(x1,x0,equal_var=False); mw=stats.mannwhitneyu(x1,x0,alternative="two-sided")
    return pd.DataFrame([{"Analysis":"Primary two-group", "Target":target,"Outcome":outcome,
        "Sample_Restriction":restriction,"Test":"Welch unequal-variance t-test","Statistic":welch.statistic,
        "P_Value":welch.pvalue,"Adjusted_P_Value":welch.pvalue,"Mean_Difference":x1.mean()-x0.mean(),
        "Median_Difference":np.median(x1)-np.median(x0),"Hedges_g":hedges_g(x1,x0),"Rank_Effect":cliff_delta(x1,x0)},
        {"Analysis":"Primary two-group","Target":target,"Outcome":outcome,"Sample_Restriction":restriction,
        "Test":"Mann-Whitney U","Statistic":mw.statistic,"P_Value":mw.pvalue,"Adjusted_P_Value":mw.pvalue,
        "Mean_Difference":x1.mean()-x0.mean(),"Median_Difference":np.median(x1)-np.median(x0),
        "Hedges_g":hedges_g(x1,x0),"Rank_Effect":cliff_delta(x1,x0)}])


def fit_two_factor(sample: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,object]:
    model=smf.ols(FORMULA,data=sample).fit(); classical=anova_lm(model,typ=3); hc3=anova_lm(model,typ=3,robust="hc3")
    total_ss=((sample[OUTCOME]-sample[OUTCOME].mean())**2).sum(); resid=classical.loc["Residual","sum_sq"]
    rows=[]
    for method,table in (("Classical Type III ANOVA",classical),("HC3 robust Type III",hc3)):
        for term,row in table.iterrows():
            ss=row.get("sum_sq",np.nan)
            rows.append({"Inference_Method":method,"Term":term,"Sum_Squares":ss,"DF":row.get("df"),
                "Statistic":row.get("F"),"P_Value":row.get("PR(>F)"),
                "Eta_Squared":ss/total_ss if np.isfinite(ss) and term!="Residual" else np.nan,
                "Partial_Eta_Squared":ss/(ss+resid) if np.isfinite(ss) and term!="Residual" else np.nan})
    hac=model.get_robustcov_results(cov_type="HAC",maxlags=5,use_correction=True)
    hac_rows=[]; ci=hac.conf_int()
    for i,name in enumerate(model.params.index):
        hac_rows.append({"Result_Type":"Coefficient","Term":name,"Estimate":model.params.iloc[i],
            "HAC_SE":hac.bse[i],"CI_Lower":ci[i,0],"CI_Upper":ci[i,1],"Statistic":hac.tvalues[i],
            "P_Value":hac.pvalues[i],"Reference_Categories":"Transition=0; Volatility=Low","HAC_Max_Lag":5})
    names=list(model.params.index); vol=[i for i,n in enumerate(names) if "Volatility_Category" in n and ":" not in n]
    inter=[i for i,n in enumerate(names) if ":" in n]
    for label,indices in (("Joint volatility-category terms",vol),("Joint transition-by-volatility interactions",inter)):
        R=np.zeros((len(indices),len(names)));
        for r,i in enumerate(indices): R[r,i]=1
        test=hac.wald_test(R,scalar=True)
        hac_rows.append({"Result_Type":"Joint_Wald_Test","Term":label,"Estimate":np.nan,"HAC_SE":np.nan,
            "CI_Lower":np.nan,"CI_Upper":np.nan,"Statistic":float(test.statistic),"P_Value":float(test.pvalue),
            "Reference_Categories":"Transition=0; Volatility=Low","HAC_Max_Lag":5})
    return pd.DataFrame(rows),pd.DataFrame(hac_rows),model


def moving_block_indices(n:int, block_length:int, rng:np.random.Generator) -> np.ndarray:
    starts=rng.integers(0,n-block_length+1,size=int(np.ceil(n/block_length)))
    return np.concatenate([np.arange(s,s+block_length) for s in starts])[:n]


def block_bootstrap(sample: pd.DataFrame, reps=2000, block_length=20, seed=12012) -> pd.DataFrame:
    rng=np.random.default_rng(seed); stats_by={k:[] for k in ["Overall","Low","Medium","High",
        "Low minus Medium interaction","Low minus High interaction","Medium minus High interaction"]}; failed=0
    def diffs(data):
        out={}
        for cat in [None]+VOL_LABELS:
            d=data if cat is None else data.loc[data.Volatility_Category.eq(cat)]
            a=d.loc[d.Transition.eq(1),OUTCOME]; b=d.loc[d.Transition.eq(0),OUTCOME]
            out["Overall" if cat is None else cat]=a.mean()-b.mean() if len(a) and len(b) else np.nan
        out["Low minus Medium interaction"]=out["Low"]-out["Medium"]
        out["Low minus High interaction"]=out["Low"]-out["High"]
        out["Medium minus High interaction"]=out["Medium"]-out["High"]
        return out
    point=diffs(sample)
    for _ in range(reps):
        values=diffs(sample.iloc[moving_block_indices(len(sample),block_length,rng)])
        if not all(np.isfinite(list(values.values()))): failed+=1; continue
        for key,value in values.items(): stats_by[key].append(value)
    return pd.DataFrame([{"Statistic":"Transition minus no-transition mean difference" if "interaction" not in k else "Difference-in-differences",
        "Sample_Restriction":"Training","Volatility_Category_or_Contrast":k,"Point_Estimate":point[k],
        "CI_Lower":np.quantile(v,.025),"CI_Upper":np.quantile(v,.975),"Requested_Replications":reps,
        "Valid_Replications":len(v),"Failed_Replications":failed,"Block_Length":block_length,"Seed":seed} for k,v in stats_by.items()])


def event_windows(frame: pd.DataFrame, radius=10, cutoff="2017-12-31") -> pd.DataFrame:
    states=frame.HMM_Filtered_State; dates=pd.to_datetime(frame.Date); cutoff=pd.Timestamp(cutoff)
    transitions=[i for i in range(1,len(frame)) if pd.notna(states.iloc[i-1]) and pd.notna(states.iloc[i]) and states.iloc[i]!=states.iloc[i-1]]
    transition_set=set(transitions); rows=[]
    for event_id,i in enumerate(transitions,1):
        origin,dest=int(states.iloc[i-1]),int(states.iloc[i]); start,end=max(0,i-radius),min(len(frame)-1,i+radius)
        others=[j for j in transition_set if start<=j<=end and j!=i]; pre=any(i-radius<=j<i for j in others)
        sample="Train" if dates.iloc[end]<=cutoff else ("Test" if dates.iloc[start]>cutoff else "Cross_Cutoff")
        kinds=[]
        if origin in {1,2} and dest in {3,4}: kinds.append("Entry_Higher_Stress")
        if origin in {3,4} and dest in {1,2}: kinds.append("Exit_Higher_Stress")
        if origin in {1,2} and dest in {1,2}: kinds.append("Within_Lower_Stress")
        if origin in {3,4} and dest in {3,4}: kinds.append("Within_Higher_Stress")
        if dest==4: kinds.append("Acute_Entry")
        if origin==4: kinds.append("Acute_Exit")
        for j in range(start,end+1):
            rows.append({"Event_ID":event_id,"Transition_Date":dates.iloc[i],"Event_Time":j-i,"Origin_State":origin,
                "Destination_State":dest,"Transition_Classification":"|".join(kinds),"Event_Sample":sample,
                "Another_Transition_In_Window":bool(others),"Another_Transition_In_Pre_Window":pre,
                "Abnormal_Volume":frame.Abnormal_Volume.iloc[j],"Log_Abnormal_Volume":frame.Log_Abnormal_Volume.iloc[j],
                "GARCH_Volatility_TrainFit":frame.GARCH_Volatility_TrainFit.iloc[j]})
    return pd.DataFrame(rows)


def event_summary(windows: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    train=windows.loc[windows.Event_Sample.eq("Train")]
    for restriction,data in (("All_Training_Events",train),("No_Other_Pre_Event_Transition",train.loc[~train.Another_Transition_In_Pre_Window])):
        for event_type in ["All"]+[x for x in ["Entry_Higher_Stress","Exit_Higher_Stress","Within_Lower_Stress","Within_Higher_Stress","Acute_Entry","Acute_Exit"]]:
            part=data if event_type=="All" else data.loc[data.Transition_Classification.str.contains(event_type)]
            for k,values in part.groupby("Event_Time"):
                x=values.Log_Abnormal_Volume.dropna(); se=x.std(ddof=1)/np.sqrt(len(x)); crit=stats.t.ppf(.975,len(x)-1) if len(x)>1 else np.nan
                rows.append({"Restriction":restriction,"Event_Type":event_type,"Event_Time":k,"Unique_Events":values.Event_ID.nunique(),
                    "Mean_Log_Abnormal_Volume":x.mean(),"Median_Log_Abnormal_Volume":x.median(),
                    "CI_Lower":x.mean()-crit*se,"CI_Upper":x.mean()+crit*se})
    return pd.DataFrame(rows)
