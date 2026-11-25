import pandas as pd
import pytest
from analytics import DataInputError, clean_dataset, load_dataset, profile_dataset, safe_csv_bytes

def test_load_csv_and_unique_headers():
    df=load_dataset("data.csv",b"a,a\n1,2\n")
    assert list(df.columns)==["a","a.1"]

def test_reject_unsupported_file():
    with pytest.raises(DataInputError): load_dataset("data.txt",b"x")

def test_profile_counts_missing_and_duplicates():
    df=pd.DataFrame({"x":[1,1,None],"y":["a","a","b"]})
    p=profile_dataset(df)
    assert p["rows"]==3 and p["missing_cells"]==1 and p["duplicate_rows"]==1

def test_cleaning_does_not_mutate_original():
    df=pd.DataFrame({"x":[1,None,1]})
    out=clean_dataset(df,drop_duplicates=True,missing_strategy="Fill numeric with median")
    assert df.isna().sum().iloc[0]==1 and out.isna().sum().iloc[0]==0

def test_csv_formula_neutralization():
    raw=safe_csv_bytes(pd.DataFrame({"text":["=1+1","safe"]})).decode("utf-8-sig")
    assert "'=1+1" in raw
