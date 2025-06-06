import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import joblib
from scipy.sparse import hstack
import numpy as np

def convert_pipe_string(s):
    try:
        parts = s.split('|')
        nums = [float(p.replace(',', '.')) for p in parts if p.strip()]
        return np.mean(nums) if nums else np.nan
    except Exception:
        return np.nan

def train_model_from_excel(file_path: str, target_col: str = 'EPT'):
    df = pd.read_excel(file_path, engine='openpyxl')

    # Pulisce le colonne con pipe e virgole
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).iloc[0]
            if '|' in sample and any(c in sample for c in [',', '.']):
                df[col] = df[col].astype(str).apply(convert_pipe_string)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Individua colonne categoriali
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [col for col in cat_cols if X[col].nunique() < 100]

    encoder = None
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        X_cat_sparse = encoder.fit_transform(X[cat_cols])

        # Dati numerici forzati float
        X_num = X.drop(columns=cat_cols).apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

        # Nomi feature
        num_cols = X_num.columns.tolist()
        enc_feat = encoder.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + enc_feat

        X = hstack([X_num.values, X_cat_sparse])
    else:
        feature_names = X.columns.tolist()
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values

    # Split e training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Assegna feature names al booster
    model.get_booster().feature_names = feature_names

    # Salvataggio
    joblib.dump(model, 'xgb_ept_model.pkl')
    if encoder:
        joblib.dump(encoder, 'ept_encoder.pkl')

    return model, encoder, score, feature_names, cat_cols


