# models/ml/train_ml.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from config.settings import model_config, path_config
from utils.metrics import calculate_trading_metrics
from loguru import logger

class MLModelTrainer:
    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.models_dir = Path(path_config.ML_MODELS_DIR)
        self.data_path = Path(path_config.PROCESSED_DATA_DIR) / f"{symbol.replace('/', '_')}_features.csv"
        
    def load_data(self) -> tuple:
        """بارگذاری داده‌های ویژگی"""
        df = pd.read_csv(self.data_path, index_col='timestamp', parse_dates=True)
        
        # جدا کردن features و labels
        feature_cols = [col for col in df.columns if not col.startswith('label_')]
        X = df[feature_cols]
        
        # استفاده از label جهت حرکت
        y = df['label_direction']
        
        return X, y, df
    
    def prepare_features(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """آماده‌سازی ویژگی‌ها برای مدل"""
        # حذف NaNها
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # تقسیم داده به train/test با حفظ ترتیب زمانی
        split_idx = int(len(X_scaled) * model_config.TRAIN_TEST_SPLIT)
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y_clean[:split_idx]
        y_test = y_clean[split_idx:]
        
        # ذخیره scaler
        scaler_path = self.models_dir / f"{self.symbol}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """آموزش مدل Random Forest"""
        logger.info("Training Random Forest...")
        
        # Grid Search برای یافتن بهترین پارامترها
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # استفاده از TimeSeriesSplit برای جلوگیری از data leakage
        tscv = TimeSeriesSplit(n_splits=5)
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, 
            scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # بهترین مدل
        best_rf = grid_search.best_estimator_
        
        # پیش‌بینی
        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
        
        # ارزیابی
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
        
        # ذخیره مدل
        model_path = self.models_dir / f"{self.symbol}_random_forest.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_rf, f)
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': [col for col in X_train.columns] if hasattr(X_train, 'columns') else list(range(X_train.shape[1])),
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return best_rf, metrics, feature_importance
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """آموزش مدل XGBoost"""
        logger.info("Training XGBoost...")
        
        # پارامترها
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        xgb_model = xgb.XGBClassifier(**params)
        
        # آموزش با validation set
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        xgb_model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # پیش‌بینی
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # ارزیابی
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        # ذخیره مدل
        model_path = self.models_dir / f"{self.symbol}_xgboost.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        
        return xgb_model, metrics
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """آموزش مدل LightGBM"""
        logger.info("Training LightGBM...")
        
        # پارامترها
        params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'metric': 'binary_logloss'
        }
        
        lgb_model = lgb.LGBMClassifier(**params)
        
        # آموزش
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # پیش‌بینی
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # ارزیابی
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "LightGBM")
        
        # ذخیره مدل
        model_path = self.models_dir / f"{self.symbol}_lightgbm.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(lgb_model, f)
        
        return lgb_model, metrics
    
    def train_catboost(self, X_train, y_train, X_test, y_test):
        """آموزش مدل CatBoost"""
        logger.info("Training CatBoost...")
        
        # CatBoost می‌تواند با مقادیر NaN کار کند
        cb_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.01,
            loss_function='Logloss',
            random_seed=42,
            verbose=False
        )
        
        # آموزش
        cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # پیش‌بینی
        y_pred = cb_model.predict(X_test)
        y_pred_proba = cb_model.predict_proba(X_test)[:, 1]
        
        # ارزیابی
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "CatBoost")
        
        # ذخیره مدل
        model_path = self.models_dir / f"{self.symbol}_catboost.pkl"
        cb_model.save_model(str(model_path))
        
        return cb_model, metrics
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """ارزیابی مدل"""
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'precision': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision'],
            'recall': classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall']
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self):
        """آموزش تمام مدل‌های ML"""
        logger.info(f"Starting ML model training for {self.symbol}...")
        
        # 1. بارگذاری داده‌ها
        X, y, df = self.load_data()
        
        # 2. آماده‌سازی ویژگی‌ها
        X_train, X_test, y_train, y_test = self.prepare_features(X, y)
        
        # 3. آموزش مدل‌ها
        results = {}
        
        # Random Forest
        rf_model, rf_metrics, feature_importance = self.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'feature_importance': feature_importance
        }
        
        # XGBoost
        xgb_model, xgb_metrics = self.train_xgboost(
            X_train, y_train, X_test, y_test
        )
        results['xgboost'] = {
            'model': xgb_model,
            'metrics': xgb_metrics
        }
        
        # LightGBM
        lgb_model, lgb_metrics = self.train_lightgbm(
            X_train, y_train, X_test, y_test
        )
        results['lightgbm'] = {
            'model': lgb_model,
            'metrics': lgb_metrics
        }
        
        # CatBoost
        cb_model, cb_metrics = self.train_catboost(
            X_train, y_train, X_test, y_test
        )
        results['catboost'] = {
            'model': cb_model,
            'metrics': cb_metrics
        }
        
        # 4. ذخیره نتایج
        self.save_results(results)
        
        logger.info("ML model training completed!")
        return results
    
    def save_results(self, results):
        """ذخیره نتایج آموزش"""
        # ذخیره metrics
        metrics_df = pd.DataFrame([r['metrics'] for r in results.values()])
        metrics_path = self.models_dir / f"{self.symbol}_ml_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"Metrics saved to {metrics_path}")
        
        # ذخیره feature importance
        if 'feature_importance' in results['random_forest']:
            fi = results['random_forest']['feature_importance']
            fi_path = self.models_dir / f"{self.symbol}_feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            
            logger.info(f"Feature importance saved to {fi_path}")

def train_all_ml_models():
    """آموزش مدل‌های ML برای تمام جفت‌ارزها"""
    trainer = MLModelTrainer('EURUSD')
    results = trainer.train_all_models()
    return results

if __name__ == "__main__":
    train_all_ml_models()