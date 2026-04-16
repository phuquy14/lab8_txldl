import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Bài 1 & 2: Xử lý Outlier bằng phương pháp IQR"""
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_ = X_df.quantile(0.25) - 1.5 * (X_df.quantile(0.75) - X_df.quantile(0.25))
        self.upper_ = X_df.quantile(0.75) + 1.5 * (X_df.quantile(0.75) - X_df.quantile(0.25))
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.lower_.index:
                X_df[col] = np.clip(X_df[col], self.lower_[col], self.upper_[col])
        return X_df.values

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Bài 1: Trích xuất đặc trưng thời gian"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_date = pd.to_datetime(pd.Series(X.iloc[:, 0]))
        return np.c_[X_date.dt.month, X_date.dt.quarter]

# BÀI 1:
def build_preprocessor(num_cols, cat_cols, text_col, date_col):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier', OutlierClipper()),
        ('scaling', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    date_pipeline = Pipeline([
        ('extractor', DateFeatureExtractor()),
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('text', TfidfVectorizer(stop_words='english', max_features=20), text_col),
        ('date', date_pipeline, date_col)
    ])
    return preprocessor

# BÀI 2:
def run_stress_test(preprocessor, df):
    print("\n--- BÀI 2: KIỂM THỬ PIPELINE ---")
    test_data = df.head(5).copy()
    test_data.loc[0, 'Neighborhood'] = 'MARS_CITY' 
    test_data.loc[1, 'LotArea'] = np.nan
    
    try:
        preprocessor.fit(df)
        output = preprocessor.transform(test_data)
        print(f"Smoke Test: PASSED. Output shape: {output.shape}")
    except Exception as e:
        print(f"Smoke Test: FAILED. Error: {e}")

# BÀI 3:
def train_and_evaluate(preprocessor, X, y):
    print("\n--- BÀI 3: ĐÁNH GIÁ MÔ HÌNH (5-FOLD CV) ---")
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf_model)
    ])

    scoring = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    cv_results = cross_validate(full_pipeline, X, y, cv=5, scoring=scoring)

    print(f"RandomForest - Average RMSE: {-cv_results['test_rmse'].mean():.2f}")
    print(f"RandomForest - Average R2: {cv_results['test_r2'].mean():.4f}")

    full_pipeline.fit(X, y)
    return full_pipeline

# BÀI 4:
def export_model(pipeline, filename='house_model.joblib'):
    joblib.dump(pipeline, filename)
    print(f"\n--- BÀI 4: Đã xuất mô hình ra file {filename} ---")

def predict_price(new_data, model_path='house_model.joblib'):
    loaded_model = joblib.load(model_path)
    preds = loaded_model.predict(new_data)
    return preds

if __name__ == "__main__":
    data = pd.read_csv('ITA105_Lab_8.csv')
    X = data.drop(columns=['SalePrice', 'ImagePath'])
    y = data['SalePrice']

    NUM_COLS = ['LotArea', 'Rooms', 'NoiseFeature', 'HasGarage']
    CAT_COLS = ['Neighborhood', 'Condition']
    TEXT_COL = 'Description'
    DATE_COL = ['SaleDate']

    my_preprocessor = build_preprocessor(NUM_COLS, CAT_COLS, TEXT_COL, DATE_COL)

    run_stress_test(my_preprocessor, data)

    final_pipe = train_and_evaluate(my_preprocessor, X, y)

    export_model(final_pipe)

    sample_input = X.head(1)
    price_pred = predict_price(sample_input)
    print(f"Dự báo giá cho nhà mẫu: {price_pred[0]:,.2f} USD")