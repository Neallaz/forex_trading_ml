"""
Improved ML Model Trainer for Forex Trading
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config.settings import settings

class ImprovedMLModelTrainer:
    """Improved ML Model Trainer with better handling"""
    
    def __init__(self):
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.models_dir = Path(settings.MODELS_DIR) / "ml"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_features(self, symbol):
        """Load engineered features"""
        file_path = self.processed_dir / f"{symbol}_features.csv"
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded {symbol} data: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        return df
    
    def analyze_data_quality(self, df):
        """Analyze data quality before training"""
        print("\n🔍 Data Quality Analysis:")
        print(f"   Total samples: {len(df)}")
        print(f"   Features: {len(df.columns) - 2}")  # minus target columns
        
        if 'target' in df.columns:
            target_dist = df['target'].value_counts()
            print(f"   Target distribution: {target_dist.to_dict()}")
            print(f"   Class ratio: {target_dist[1]/len(df):.3f}")
        
        # Check for NaN
        nan_counts = df.isna().sum()
        if nan_counts.any():
            print(f"   ⚠️ NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        
        return df
    
    def prepare_data_with_balance(self, df):
        """Prepare data with proper balancing"""
        print("\n📊 Preparing data...")
        
        if 'target' not in df.columns:
            print("❌ 'target' column not found!")
            return None
        
        X = df.drop(['target', 'target_return'], axis=1, errors='ignore')
        y = df['target'].astype(int)
        
        # Time-based split (مهم برای داده‌های زمانی)
        split_idx = int(len(X) * settings.TRAIN_TEST_SPLIT)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"   Test class distribution: {y_test.value_counts().to_dict()}")
        
        # Handle imbalance if severe
        train_class_ratio = y_train.value_counts().min() / len(y_train)
        
        if train_class_ratio < 0.3:  # اگر کلاس اقلیت کمتر از ۳۰٪ باشه
            print(f"   ⚠️ Severe imbalance detected ({train_class_ratio:.1%}). Applying SMOTE...")
            
            # فقط اگر به اندازه کافی نمونه داریم
            if y_train.value_counts().min() > 10:  # حداقل ۱۰ نمونه از کلاس اقلیت
                smote = SMOTE(
                    random_state=settings.RANDOM_SEED,
                    k_neighbors=min(5, y_train.value_counts().min() - 1)
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"   ✅ After SMOTE - Train: {pd.Series(y_train).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest_improved(self, X_train, y_train):
        """Improved Random Forest for forex"""
        print("\n🌲 Training Random Forest...")
        
        # تنظیمات بهینه برای داده‌های فارکس
        rf = RandomForestClassifier(
            n_estimators=150,  # کم‌تر برای جلوگیری از overfitting
            max_depth=12,      # عمق متوسط
            min_samples_split=20,  # بیشتر برای داده‌های نویزی
            min_samples_leaf=10,
            max_features='sqrt',  # بهتر از 'auto' برای داده‌های با featureهای زیاد
            bootstrap=True,
            oob_score=True,  # برای validation داخلی
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced_subsample',  # بهترین گزینه برای imbalance
            verbose=0
        )
        
        rf.fit(X_train, y_train)
        print(f"   OOB Score: {rf.oob_score_:.3f}")
        return rf
    
    def train_xgboost_improved(self, X_train, y_train):
        """Improved XGBoost for forex"""
        print("\n📈 Training XGBoost...")
        
        # محاسبه وزن برای کلاس اقلیت
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        scale_pos_weight = min(scale_pos_weight, 10)  # محدود کردن به حداکثر ۱۰
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,  # کم‌تر برای جلوگیری از overfitting
            learning_rate=0.05,
            subsample=0.8,  # برای جلوگیری از overfitting
            colsample_bytree=0.8,
            gamma=1,  # regularization
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1,   # L2 regularization
            scale_pos_weight=scale_pos_weight,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        xgb.fit(X_train, y_train)
        return xgb
    
    def train_lightgbm_improved(self, X_train, y_train):
        """Improved LightGBM for forex"""
        print("\n💡 Training LightGBM...")
        
        # محاسبه وزن کلاس
        class_weights = len(y_train) / (2 * np.bincount(y_train))
        
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,  # محدود کردن برای جلوگیری از overfitting
            min_child_samples=20,  # بیشتر برای داده‌های نویزی
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            verbose=-1  # غیرفعال کردن warnings
        )
        
        lgbm.fit(X_train, y_train)
        return lgbm
    
    def train_logistic_regression_improved(self, X_train, y_train):
        """Improved Logistic Regression for forex"""
        print("\n📉 Training Logistic Regression...")
        
        lr = LogisticRegression(
            C=0.1,  # قوی‌تر regularization
            max_iter=2000,
            class_weight='balanced',
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            solver='saga',  # بهتر برای داده‌های بزرگ
            penalty='elasticnet',  # ترکیب L1 و L2
            l1_ratio=0.5
        )
        
        lr.fit(X_train, y_train)
        return lr
    
    def evaluate_model_improved(self, model, X_test, y_test, model_name):
        """Comprehensive evaluation"""
        print(f"\n{'='*40}")
        print(f"Evaluating {model_name}")
        print(f"{'='*40}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Add ROC-AUC if possible
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        print(f"📊 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"🎯 Precision: {metrics['precision']:.4f}")
        print(f"↩️ Recall:    {metrics['recall']:.4f}")
        print(f"⭐ F1-Score:  {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"📈 ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🧮 Confusion Matrix:")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return metrics, y_pred_proba
    
    def train_all_models_improved(self, symbol):
        """Train all improved ML models"""
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING IMPROVED ML MODELS FOR {symbol}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_features(symbol)
        if df is None:
            return None
        
        # Analyze data
        df = self.analyze_data_quality(df)
        
        # Prepare data with balancing
        data = self.prepare_data_with_balance(df)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test = data
        
        # Train models
        models = {}
        predictions = {}
        metrics_list = []
        
        # 1. Random Forest
        rf_model = self.train_random_forest_improved(X_train, y_train)
        models['random_forest'] = rf_model
        rf_metrics, rf_preds = self.evaluate_model_improved(
            rf_model, X_test, y_test, "Random Forest"
        )
        metrics_list.append({**rf_metrics, 'model': 'random_forest'})
        predictions['random_forest'] = rf_preds
        
        # 2. XGBoost
        xgb_model = self.train_xgboost_improved(X_train, y_train)
        models['xgboost'] = xgb_model
        xgb_metrics, xgb_preds = self.evaluate_model_improved(
            xgb_model, X_test, y_test, "XGBoost"
        )
        metrics_list.append({**xgb_metrics, 'model': 'xgboost'})
        predictions['xgboost'] = xgb_preds
        
        # 3. LightGBM
        lgbm_model = self.train_lightgbm_improved(X_train, y_train)
        models['lightgbm'] = lgbm_model
        lgbm_metrics, lgbm_preds = self.evaluate_model_improved(
            lgbm_model, X_test, y_test, "LightGBM"
        )
        metrics_list.append({**lgbm_metrics, 'model': 'lightgbm'})
        predictions['lightgbm'] = lgbm_preds
        
        # 4. Logistic Regression
        lr_model = self.train_logistic_regression_improved(X_train, y_train)
        models['logistic_regression'] = lr_model
        lr_metrics, lr_preds = self.evaluate_model_improved(
            lr_model, X_test, y_test, "Logistic Regression"
        )
        metrics_list.append({**lr_metrics, 'model': 'logistic_regression'})
        predictions['logistic_regression'] = lr_preds
        
        # Save everything
        self.save_results(symbol, models, metrics_list, predictions, X_test, y_test)
        
        return {
            'models': models,
            'metrics': pd.DataFrame(metrics_list),
            'predictions': pd.DataFrame(predictions, index=X_test.index),
            'X_test': X_test,
            'y_test': y_test
        }
    
    def save_results(self, symbol, models, metrics_list, predictions, X_test, y_test):
        """Save all results"""
        print(f"\n💾 Saving results for {symbol}...")
        
        # Save models
        for model_name, model in models.items():
            model_path = self.models_dir / f"{symbol}_{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"   ✅ Model saved: {model_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics_list)
        results_path = self.models_dir / f"{symbol}_ml_results.csv"
        metrics_df.to_csv(results_path, index=False)
        print(f"   ✅ Metrics saved: {results_path}")
        
        # Save predictions
        preds_df = pd.DataFrame(predictions, index=X_test.index)
        preds_df['actual'] = y_test.values
        preds_path = self.models_dir / f"{symbol}_predictions.csv"
        preds_df.to_csv(preds_path)
        print(f"   ✅ Predictions saved: {preds_path}")
        
        # Print summary
        print(f"\n📋 Summary for {symbol}:")
        print(metrics_df.to_string(index=False))
    
    def train_for_all_pairs(self):
        """Train for all currency pairs"""
        all_results = {}
        
        print(f"\n{'='*60}")
        print(f"🎯 STARTING ML TRAINING FOR ALL PAIRS")
        print(f"{'='*60}")
        
        for pair in settings.FOREX_PAIRS[:2]:  # با ۲ جفت ارز شروع کن
            print(f"\n{'='*60}")
            print(f"PROCESSING: {pair}")
            print(f"{'='*60}")
            
            results = self.train_all_models_improved(pair)
            if results:
                all_results[pair] = results
                print(f"\n✅ Successfully trained {pair}")
            else:
                print(f"\n❌ Failed to train {pair}")
        
        return all_results

if __name__ == "__main__":
    trainer = ImprovedMLModelTrainer()
    results = trainer.train_for_all_pairs()
    
    print(f"\n{'='*60}")
    print("🎉 ML MODEL TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total pairs trained: {len(results)}")
    
# """
# Train Classic Machine Learning Models
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# import warnings
# warnings.filterwarnings('ignore')
# import os
# import sys
# # Add project root to Python path
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, PROJECT_ROOT)

# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import classification_report, confusion_matrix

# from config.settings import settings

# class MLModelTrainer:
#     """Machine Learning Model Trainer Class"""
    
#     def __init__(self):
#         self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
#         self.models_dir = Path(settings.MODELS_DIR) / "ml"
#         self.models_dir.mkdir(parents=True, exist_ok=True)
        
#     def load_features(self, symbol):
#         """Load engineered features"""
#         file_path = self.processed_dir / f"{symbol}_features.csv"
#         if not file_path.exists():
#             return None
        
#         df = pd.read_csv(file_path, index_col=0, parse_dates=True)
#         return df
    
#     def prepare_data(self, df):
#         """Prepare data for training"""
#         # Separate features and target
#         X = df.drop(['target', 'target_return'], axis=1)
#         y = df['target'].astype(int)  # classification target
        
#         # Time-based split
#         split_idx = int(len(X) * settings.TRAIN_TEST_SPLIT)
        
#         X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#         y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
#         return X_train, X_test, y_train, y_test
    
#     def train_random_forest(self, X_train, y_train):
#         """Train Random Forest model"""
#         print("Training Random Forest...")
        
#         # Parameters for tuning
#         param_grid = {
#             'n_estimators': [100, 200],
#             'max_depth': [10, 20, None],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
        
#         # Use TimeSeriesSplit for cross-validation
#         tscv = TimeSeriesSplit(n_splits=5)
        
#         # Base model
#         rf = RandomForestClassifier(
#             random_state=settings.RANDOM_SEED,
#             n_jobs=-1,
#             class_weight='balanced'
#         )
        
#         # Grid Search
#         grid_search = GridSearchCV(
#             rf, param_grid, 
#             cv=tscv, 
#             scoring='f1',
#             n_jobs=-1,
#             verbose=1
#         )
        
#         grid_search.fit(X_train, y_train)
        
#         print(f"Best parameters: {grid_search.best_params_}")
#         print(f"Best score: {grid_search.best_score_:.4f}")
        
#         return grid_search.best_estimator_
    
    
    
#     def train_lightgbm(self, X_train, y_train):
#         """Train LightGBM model"""
#         print("Training LightGBM...")
        
#         lgbm = LGBMClassifier(
#             n_estimators=200,
#             max_depth=7,
#             learning_rate=0.05,
#             random_state=settings.RANDOM_SEED,
#             n_jobs=-1
#         )
        
#         lgbm.fit(X_train, y_train)
#         return lgbm
    
#     def train_logistic_regression(self, X_train, y_train):
#         """Train Logistic Regression model"""
#         print("Training Logistic Regression...")
        
#         lr = LogisticRegression(
#             random_state=settings.RANDOM_SEED,
#             max_iter=1000,
#             class_weight='balanced',
#             n_jobs=-1
#         )
        
#         lr.fit(X_train, y_train)
#         return lr
    
#     def evaluate_model(self, model, X_test, y_test, model_name):
#         """Evaluate model performance"""
#         y_pred = model.predict(X_test)
#         y_pred_proba = model.predict_proba(X_test)[:, 1]
        
#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred),
#             'recall': recall_score(y_test, y_pred),
#             'f1': f1_score(y_test, y_pred),
#             'model': model_name
#         }
        
#         print(f"\nEvaluation of {model_name}:")
#         print(f"Accuracy: {metrics['accuracy']:.4f}")
#         print(f"Precision: {metrics['precision']:.4f}")
#         print(f"Recall: {metrics['recall']:.4f}")
#         print(f"F1-Score: {metrics['f1']:.4f}")
        
#         # Show confusion matrix
#         cm = confusion_matrix(y_test, y_pred)
#         print(f"Confusion Matrix:\n{cm}")
        
#         # Classification report
#         print(f"\nClassification Report:")
#         print(classification_report(y_test, y_pred))
        
#         return metrics, y_pred_proba
    
#     def train_all_models(self, symbol):
#         """Train all ML models for a currency pair"""
#         print(f"\n{'='*50}")
#         print(f"Training ML models for {symbol}")
#         print(f"{'='*50}")
        
#         # Load data
#         df = self.load_features(symbol)
#         if df is None:
#             print(f"No data found for {symbol}")
#             return None
        
#         # Prepare data
#         X_train, X_test, y_train, y_test = self.prepare_data(df)
#         print(f"Sample counts - Train: {len(X_train)}, Test: {len(X_test)}")
        
#         # Train models
#         models = {}
#         predictions = {}
#         metrics_list = []
        
#         # Random Forest
#         rf_model = self.train_random_forest(X_train, y_train)
#         models['random_forest'] = rf_model
#         rf_metrics, rf_preds = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
#         metrics_list.append(rf_metrics)
#         predictions['random_forest'] = rf_preds
        
#         # XGBoost
#         xgb_model = self.train_xgboost(X_train, y_train)
#         models['xgboost'] = xgb_model
#         xgb_metrics, xgb_preds = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
#         metrics_list.append(xgb_metrics)
#         predictions['xgboost'] = xgb_preds
        
#         # LightGBM
#         lgbm_model = self.train_lightgbm(X_train, y_train)
#         models['lightgbm'] = lgbm_model
#         lgbm_metrics, lgbm_preds = self.evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
#         metrics_list.append(lgbm_metrics)
#         predictions['lightgbm'] = lgbm_preds
        
#         # Logistic Regression (baseline)
#         lr_model = self.train_logistic_regression(X_train, y_train)
#         models['logistic_regression'] = lr_model
#         lr_metrics, lr_preds = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
#         metrics_list.append(lr_metrics)
#         predictions['logistic_regression'] = lr_preds
        
#         # Save models
#         for model_name, model in models.items():
#             model_path = self.models_dir / f"{symbol}_{model_name}.pkl"
#             joblib.dump(model, model_path)
#             print(f"Model {model_name} saved to {model_path}")
        
#         # Save results
#         results_df = pd.DataFrame(metrics_list)
#         results_path = self.models_dir / f"{symbol}_ml_results.csv"
#         results_df.to_csv(results_path)
        
#         # Save predictions
#         preds_df = pd.DataFrame(predictions, index=X_test.index)
#         preds_df['actual'] = y_test.values
#         preds_path = self.models_dir / f"{symbol}_predictions.csv"
#         preds_df.to_csv(preds_path)
        
#         print(f"\nTraining results for {symbol} saved")
        
#         return {
#             'models': models,
#             'metrics': results_df,
#             'predictions': preds_df,
#             'X_test': X_test,
#             'y_test': y_test
#         }
    
#     def train_for_all_pairs(self):
#         """Train models for all currency pairs"""
#         all_results = {}
        
#         for pair in settings.FOREX_PAIRS[:2]:  # Start with two pairs
#             results = self.train_all_models(pair)
#             if results:
#                 all_results[pair] = results
        
#         return all_results

# if __name__ == "__main__":
#     trainer = MLModelTrainer()
#     results = trainer.train_for_all_pairs()
#     print(f"ML model training completed for {len(results)} pairs")