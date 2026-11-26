"""Local predictive modeling and anomaly detection."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def _prep(x):
    nums=x.select_dtypes(include=np.number).columns.tolist()
    cats=[c for c in x.columns if c not in nums]
    parts=[]
    if nums: parts.append(("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats: parts.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",max_categories=50))]),cats))
    return ColumnTransformer(parts)

def train_predictive_model(df,target,task="Auto-detect",test_size=.2,random_state=42):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Load a non-empty dataset before training a model.")
    if df.columns.duplicated().any():
        raise ValueError("Model training requires unique column names. Rename duplicate columns first.")
    if target not in df:
        raise ValueError("Target column does not exist.")
    if not isinstance(test_size, (int, float)) or not 0 < float(test_size) < 1:
        raise ValueError("Test-set share must be greater than 0 and less than 1.")
    data=df.replace([np.inf,-np.inf],np.nan).dropna(subset=[target]).copy()
    y=data[target]
    if len(data)<20:
        raise ValueError("At least 20 non-missing target rows are required.")
    if task=="Auto-detect":
        task="Regression" if pd.api.types.is_numeric_dtype(y) and y.nunique()>10 else "Classification"
    if task not in {"Regression","Classification"}:
        raise ValueError("Choose Regression or Classification.")
    if task=="Classification" and not 2<=y.nunique()<=30:
        raise ValueError("Classification requires 2–30 target classes.")
    x=data.drop(columns=[target]).dropna(axis=1,how="all")
    high=[c for c in x.select_dtypes(exclude=np.number) if x[c].nunique()>min(200,len(x)*.5)]
    x=x.drop(columns=high)
    constant=[c for c in x.columns if x[c].nunique(dropna=True)<=1]
    x=x.drop(columns=constant)
    if x.empty:
        raise ValueError("No varying, usable feature columns remain after removing empty, constant and high-cardinality fields.")
    strat=None
    if task=="Classification":
        class_counts=y.value_counts(dropna=False)
        class_count=len(class_counts)
        if class_counts.min()<2:
            raise ValueError("Each classification target class needs at least two observations. Add more examples or combine justified rare classes.")
        n_test=int(np.ceil(len(y)*float(test_size)))
        n_train=len(y)-n_test
        if n_test<class_count or n_train<class_count:
            raise ValueError(
                f"The selected test split is too small for {class_count} target classes. "
                "Increase the test-set share or provide more observations per class."
            )
        strat=y
    xt,xv,yt,yv=train_test_split(
        x, y, test_size=float(test_size), random_state=random_state, stratify=strat
    )
    if task=="Regression":
        model=RandomForestRegressor(n_estimators=120,max_depth=12,min_samples_leaf=2,random_state=random_state,n_jobs=-1)
        pipe=Pipeline([("prep",_prep(x)),("model",model)]).fit(xt,yt); pred=pipe.predict(xv)
        metrics={"MAE":float(mean_absolute_error(yv,pred)),"RMSE":float(np.sqrt(mean_squared_error(yv,pred))),"R²":float(r2_score(yv,pred))}
    else:
        enc=LabelEncoder(); yy=enc.fit_transform(y.astype(str)); yt=pd.Series(yy,index=data.index).loc[xt.index]; yv=pd.Series(yy,index=data.index).loc[xv.index]
        model=RandomForestClassifier(n_estimators=120,max_depth=12,min_samples_leaf=2,class_weight="balanced",random_state=random_state,n_jobs=-1)
        pipe=Pipeline([("prep",_prep(x)),("model",model)]).fit(xt,yt); pred=pipe.predict(xv)
        metrics={"Accuracy":float(accuracy_score(yv,pred)),"F1 weighted":float(f1_score(yv,pred,average="weighted",zero_division=0))}
        yv=enc.inverse_transform(yv); pred=enc.inverse_transform(pred)
    return {"task":task,"metrics":metrics,"comparison":pd.DataFrame({"actual":np.asarray(yv),"predicted":np.asarray(pred)}),"model":pipe,"features":x.columns.tolist(),"dropped_high_cardinality":high}

def detect_anomalies(df,contamination=.05):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Load a non-empty dataset before detecting anomalies.")
    if not isinstance(contamination, (int, float)) or not 0 < float(contamination) <= 0.5:
        raise ValueError("Expected anomaly share must be greater than 0 and no more than 0.5.")
    reserved = {"anomaly_flag", "anomaly_score"} & set(df.columns)
    if reserved:
        raise ValueError(
            "The dataset already contains reserved anomaly output column(s): "
            + ", ".join(sorted(reserved)) + ". Rename them before running anomaly detection."
        )
    n = df.select_dtypes(include=np.number)
    if n.shape[1] < 1 or len(n) < 10:
        raise ValueError("Anomaly detection requires at least one numeric column and 10 rows.")
    numeric = n.replace([np.inf, -np.inf], np.nan)
    varying_columns = [
        column for column in numeric.columns
        if numeric[column].nunique(dropna=True) > 1
    ]
    if not varying_columns:
        raise ValueError(
            "Anomaly detection requires at least one numeric feature with variation. "
            "Constant and entirely missing columns cannot identify anomalies."
        )
    x = numeric[varying_columns].fillna(numeric[varying_columns].median()).fillna(0)
    model=IsolationForest(contamination=contamination,random_state=42,n_estimators=150)
    labels=model.fit_predict(x); out=df.copy()
    out["anomaly_flag"]=np.where(labels==-1,"Anomaly","Normal"); out["anomaly_score"]=model.decision_function(x)
    return out
