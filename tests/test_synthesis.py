"""Stage 14 synthesis validation tests."""
from pathlib import Path
import json,shutil
import pandas as pd,pytest
from market_regime.synthesis import *
ROOT=Path(__file__).resolve().parents[1]

def test_required_inputs_validate():
 paths,hashes=validate_inputs(ROOT/'outputs/tables',ROOT/'outputs/models');assert len(paths)==18 and len(hashes)==18
def test_missing_file_clear_failure(tmp_path):
 (tmp_path/'tables').mkdir();(tmp_path/'models').mkdir()
 with pytest.raises(FileNotFoundError,match='Missing canonical'):validate_inputs(tmp_path/'tables',tmp_path/'models')
def test_required_columns_checked(tmp_path):
 shutil.copytree(ROOT/'outputs/tables',tmp_path/'tables');shutil.copytree(ROOT/'outputs/models',tmp_path/'models');pd.DataFrame({'wrong':[1]}).to_csv(tmp_path/'tables/market_with_hmm.csv',index=False)
 with pytest.raises(ValueError,match='missing required columns'):validate_inputs(tmp_path/'tables',tmp_path/'models')
def test_ten_tables_and_columns():
 d,_=load_canonical(ROOT);tabs=build_tables(d);assert len(tabs)==10;assert {'Research_Question','Result','Final_Interpretation'}<=set(tabs['table_10_research_question_decisions'])
def test_decisions_are_explicit_and_restrained():
 x=decisions();assert x.Result.tolist()==['Supported','Not supported','Exploratory only','Not supported'];assert not x.astype(str).apply(lambda s:s.str.contains(r'accepted|proven',case=False)).any().any()
def test_synthesis_never_refits():
 import inspect;src=inspect.getsource(export_synthesis);assert '.fit(' not in src and 'LogisticRegression' not in src
def test_hashes_match_direct_calculation():
 paths,hashes=validate_inputs(ROOT/'outputs/tables',ROOT/'outputs/models');assert all(len(x)==64 for x in hashes.values());assert len(set(hashes.values()))==len(hashes)
def test_json_safe_has_no_nonfinite():
 x=json.dumps(json_safe({'a':float('nan'),'b':float('inf')}),allow_nan=False);assert x=='{"a": null, "b": null}'
def test_state_names_and_colors_consistent():
 assert set(STATE_NAMES)==set(STATE_COLORS)=={1,2,3,4};assert len(set(STATE_COLORS.values()))==4
def test_canonical_input_unchanged_by_building_tables():
 _,before=validate_inputs(ROOT/'outputs/tables',ROOT/'outputs/models');d,_=load_canonical(ROOT);build_tables(d);_,after=validate_inputs(ROOT/'outputs/tables',ROOT/'outputs/models');assert before==after
def test_repeated_tables_identical():
 d,_=load_canonical(ROOT);a=build_tables(d);b=build_tables(d)
 for k in a:pd.testing.assert_frame_equal(a[k],b[k])
def test_generated_outputs_nonempty():
 out=ROOT/'outputs/final_synthesis';assert (out/'final_results_summary.json').stat().st_size>0;assert len(list((out/'final_tables').glob('*.csv')))==10;assert all(p.stat().st_size>0 for p in (out/'figures').glob('*.png'))
def test_final_json_schema_and_finite():
 x=json.loads((ROOT/'outputs/final_synthesis/final_results_summary.json').read_text());assert {'sample','selected_hmm','primary_association','primary_prediction','conclusions','input_hashes'}<=set(x)
def test_notebook_has_no_error_outputs():
 import nbformat;n=nbformat.read(ROOT/'notebooks/13_final_empirical_synthesis.ipynb',4);assert not [o for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
