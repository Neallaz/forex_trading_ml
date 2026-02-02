"""
آموزش مدل‌های کلاسیک Machine Learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from config.settings import settings

class MLModelTrainer:
    """کلاس آموزش مدل‌های ML"""
    
    def __init__(self):
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.models_dir = Path(settings.MODELS_DIR) / "ml"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_features(self, symbol):
        """بارگذاری ویژگی‌های مهندسی شده"""
        file_path = self.processed_dir / f"{symbol}_features.csv"
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    
    def prepare_data(self, df):
        """آماده‌سازی داده‌ها برای آموزش"""
        # جداسازی features و target
        X = df.drop(['target', 'target_return'], axis=1)
        y = df['target'].astype(int)  # classification target
        
        # تقسیم زمانی (Time Series Split)
        split_idx = int(len(X) * settings.TRAIN_TEST_SPLIT)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """آموزش Random Forest"""
        print("آموزش Random Forest...")
        
        # پارامترهای برای tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # استفاده از TimeSeriesSplit برای cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # مدل پایه
        rf = RandomForestClassifier(
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Grid Search
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=tscv, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"بهترین پارامترها: {grid_search.best_params_}")
        print(f"بهترین امتیاز: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train):
        """آموزش XGBoost"""
        print("آموزش XGBoost...")
        
        # پارامترها
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        xgb = XGBClassifier(
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        grid_search = GridSearchCV(
            xgb, param_grid,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"بهترین پارامترهای XGBoost: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_lightgbm(self, X_train, y_train):
        """آموزش LightGBM"""
        print("آموزش LightGBM...")
        
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1
        )
        
        lgbm.fit(X_train, y_train)
        return lgbm
    
    def train_logistic_regression(self, X_train, y_train):
        """آموزش Logistic Regression"""
        print("آموزش Logistic Regression...")
        
        lr = LogisticRegression(
            random_state=settings.RANDOM_SEED,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        )
        
        lr.fit(X_train, y_train)
        return lr
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """ارزیابی مدل"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model': model_name
        }
        
        print(f"\nارزیابی {model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        
        # نمایش confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics, y_pred_proba
    
    def train_all_models(self, symbol):
        """آموزش تمام مدل‌های ML برای یک جفت ارز"""
        print(f"\n{'='*50}")
        print(f"آموزش مدل‌های ML برای {symbol}")
        print(f"{'='*50}")
        
        # بارگذاری داده‌ها
        df = self.load_features(symbol)
        if df is None:
            print(f"داده‌ای برای {symbol} یافت نشد")
            return None
        
        # آماده‌سازی داده‌ها
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        print(f"تعداد نمونه‌ها - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # آموزش مدل‌ها
        models = {}
        predictions = {}
        metrics_list = []
        
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        models['random_forest'] = rf_model
        rf_metrics, rf_preds = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        metrics_list.append(rf_metrics)
        predictions['random_forest'] = rf_preds
        
        # XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        models['xgboost'] = xgb_model
        xgb_metrics, xgb_preds = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        metrics_list.append(xgb_metrics)
        predictions['xgboost'] = xgb_preds
        
        # LightGBM
        lgbm_model = self.train_lightgbm(X_train, y_train)
        models['lightgbm'] = lgbm_model
        lgbm_metrics, lgbm_preds = self.evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
        metrics_list.append(lgbm_metrics)
        predictions['lightgbm'] = lgbm_preds
        
        # Logistic Regression (baseline)
        lr_model = self.train_logistic_regression(X_train, y_train)
        models['logistic_regression'] = lr_model
        lr_metrics, lr_preds = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        metrics_list.append(lr_metrics)
        predictions['logistic_regression'] = lr_preds
        
        # ذخیره مدل‌ها
        for model_name, model in models.items():
            model_path = self.models_dir / f"{symbol}_{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"مدل {model_name} در {model_path} ذخیره شد")
        
        # ذخیره نتایج
        results_df = pd.DataFrame(metrics_list)
        results_path = self.models_dir / f"{symbol}_ml_results.csv"
        results_df.to_csv(results_path)
        
        # ذخیره predictions
        preds_df = pd.DataFrame(predictions, index=X_test.index)
        preds_df['actual'] = y_test.values
        preds_path = self.models_dir / f"{symbol}_predictions.csv"
        preds_df.to_csv(preds_path)
        
        print(f"\nنتایج آموزش برای {symbol} ذخیره شد")
        
        return {
            'models': models,
            'metrics': results_df,
            'predictions': preds_df,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def train_for_all_pairs(self):
        """آموزش مدل‌ها برای تمام جفت‌ارزها"""
        all_results = {}
        
        for pair in settings.FOREX_PAIRS[:2]:  # برای شروع دو جفت
            results = self.train_all_models(pair)
            if results:
                all_results[pair] = results
        
        return all_results

if __name__ == "__main__":
    trainer = MLModelTrainer()
    results = trainer.train_for_all_pairs()
    print(f"آموزش مدل‌های ML برای {len(results)} جفت کامل شد")