import pandas as pd
from modeling import detect_anomalies, train_predictive_model

def test_anomaly_output_columns():
    df=pd.DataFrame({"x":list(range(30)),"y":list(range(30,60))})
    out=detect_anomalies(df)
    assert {"anomaly_flag","anomaly_score"}.issubset(out.columns)

def test_regression_model_returns_metrics():
    df=pd.DataFrame({"x":range(50),"z":[i%7 for i in range(50)],"target":[2*i+1 for i in range(50)]})
    result=train_predictive_model(df,"target","Regression")
    assert "RMSE" in result["metrics"] and len(result["comparison"])>0
