"""Generate the final Stage 14 empirical synthesis from canonical saved outputs."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from market_regime.synthesis import export_synthesis
def main():
 tables,summary,figures,validation=export_synthesis(ROOT)
 print(json.dumps({"status":"complete","tables":len(tables),"figures":len(figures),"models_refit":False,"rq2_prediction":"Not supported"},indent=2))
if __name__=="__main__":main()
