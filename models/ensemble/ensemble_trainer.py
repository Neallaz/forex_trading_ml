"""
Ensemble مدل‌های ML و DL
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from config.settings import settings

class EnsembleTrainer:
    """کلاس Ensemble مدل‌ها"""
    
    def __init__(self):
        self.models_dir = Path(settings.MODELS_DIR)
        self.ml_dir = self.models_dir / "ml"
        self.dl_dir = self.models_dir / "dl"
        self.ensemble_dir = self.models_dir / "ensemble"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models_and_predictions(self, symbol):
        """بارگذاری مدل‌ها و پیش‌بینی‌های قبلی"""
        
        # بارگذاری مدل‌های ML
        ml_models = {}
        ml_predictions = {}
        
        ml_model_files = list(self.ml_dir.glob(f"{symbol}_*.pkl"))
        for model_file in ml_model_files:
            model_name = model_file.stem.replace(f"{symbol}_", "")
            if model_name != "ml_results":
                ml_models[model_name] = joblib.load(model_file)
        
        # بارگذاری پیش‌بینی‌های ML
        ml_preds_path = self.ml_dir / f"{symbol}_predictions.csv"
        if ml_preds_path.exists():
            ml_predictions = pd.read_csv(ml_preds_path, index_col=0, parse_dates=True)
        
        # بارگذاری پیش‌بینی‌های DL
        dl_predictions = {}
        dl_preds_path = self.dl_dir / f"{symbol}_dl_predictions.csv"
        if dl_preds_path.exists():
            dl_predictions = pd.read_csv(dl_preds_path, index_col=0)
        
        return ml_models, ml_predictions, dl_predictions
    
    def create_feature_matrix(self, ml_predictions, dl_predictions):
        """ایجاد ماتریس ویژگی از پیش‌بینی‌های مدل‌ها"""
        features = {}
        
        # اضافه کردن پیش‌بینی‌های ML
        for col in ml_predictions.columns:
            if col != 'actual':
                features[f"ml_{col}"] = ml_predictions[col].values
        
        # اضافه کردن پیش‌بینی‌های DL
        for col in dl_predictions.columns:
            if col != 'actual':
                features[f"dl_{col}"] = dl_predictions[col].values
        
        # تبدیل به DataFrame
        features_df = pd.DataFrame(features)
        
        # target
        if 'actual' in ml_predictions.columns:
            y = ml_predictions['actual'].values
        else:
            y = dl_predictions['actual'].values
        
        return features_df, y
    
    def create_weighted_ensemble(self, predictions, weights=None):
        """ایجاد ensemble وزن‌دار"""
        if weights is None:
            # وزن‌دهی بر اساس دقت (فرضی)
            weights = {
                'random_forest': 0.3,
                'xgboost': 0.3,
                'lstm': 0.2,
                'attention': 0.2
            }
        
        weighted_sum = np.zeros(len(predictions))
        
        for model_name, weight in weights.items():
            if model_name in predictions.columns:
                weighted_sum += predictions[model_name] * weight
        
        return weighted_sum / sum(weights.values())
    
    def create_voting_ensemble(self, ml_models, X, y):
        """ایجاد Voting Ensemble"""
        
        estimators = [
            (name, model) for name, model in ml_models.items()
        ]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # soft voting برای probability
            n_jobs=-1
        )
        
        # آموزش ensemble
        voting_clf.fit(X, y)
        
        return voting_clf
    
    def create_stacking_ensemble(self, ml_models, X, y):
        """ایجاد Stacking Ensemble"""
        
        estimators = [
            (name, model) for name, model in ml_models.items()
        ]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        
        stacking_clf.fit(X, y)
        
        return stacking_clf
    
    def calibrate_predictions(self, predictions, y_true):
        """Calibrate کردن احتمالات"""
        calibrated = CalibratedClassifierCV(
            base_estimator=LogisticRegression(),
            cv=5
        )
        
        # باید reshape کنیم
        X_calib = predictions.values.reshape(-1, 1)
        calibrated.fit(X_calib, y_true)
        
        return calibrated
    
    def train_ensemble(self, symbol):
        """آموزش ensemble برای یک جفت ارز"""
        print(f"\n{'='*50}")
        print(f"آموزش Ensemble برای {symbol}")
        print(f"{'='*50}")
        
        # بارگذاری مدل‌ها و پیش‌بینی‌ها
        ml_models, ml_preds, dl_preds = self.load_models_and_predictions(symbol)
        
        if ml_preds.empty or dl_preds.empty:
            print("پیش‌بینی‌های کافی برای ensemble وجود ندارد")
            return None
        
        # ایجاد ماتریس ویژگی
        X_ensemble, y = self.create_feature_matrix(ml_preds, dl_preds)
        
        print(f"ویژگی‌های Ensemble: {X_ensemble.shape}")
        print(f"مدل‌های موجود: {list(ml_models.keys())}")
        
        # 1. Weighted Ensemble
        print("\n1. ایجاد Weighted Ensemble...")
        weights = {
            'random_forest': 0.25,
            'xgboost': 0.25,
            'lightgbm': 0.15,
            'logistic_regression': 0.10,
            'lstm': 0.15,
            'attention': 0.10
        }
        
        # فقط مدل‌های موجود را استفاده می‌کنیم
        available_weights = {
            k: v for k, v in weights.items() 
            if k in ml_preds.columns or f"dl_{k}" in dl_preds.columns
        }
        
        # نرمال‌سازی وزن‌ها
        total_weight = sum(available_weights.values())
        available_weights = {k: v/total_weight for k, v in available_weights.items()}
        
        weighted_pred = self.create_weighted_ensemble(
            pd.concat([ml_preds, dl_preds], axis=1),
            available_weights
        )
        
        # 2. Voting Ensemble (فقط برای مدل‌های ML)
        print("\n2. ایجاد Voting Ensemble...")
        
        # برای Voting به X اصلی نیاز داریم
        features_path = Path(settings.PROCESSED_DATA_DIR) / f"{symbol}_features.csv"
        df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        X_original = df_features.drop(['target', 'target_return'], axis=1)
        y_original = df_features['target'].astype(int)
        
        split_idx = int(len(X_original) * settings.TRAIN_TEST_SPLIT)
        X_test_original = X_original.iloc[split_idx:]
        
        voting_clf = self.create_voting_ensemble(
            ml_models,
            X_original.iloc[:split_idx],
            y_original.iloc[:split_idx]
        )
        
        voting_pred = voting_clf.predict_proba(X_test_original)[:, 1]
        
        # 3. Stacking Ensemble
        print("\n3. ایجاد Stacking Ensemble...")
        stacking_clf = self.create_stacking_ensemble(
            ml_models,
            X_original.iloc[:split_idx],
            y_original.iloc[:split_idx]
        )
        
        stacking_pred = stacking_clf.predict_proba(X_test_original)[:, 1]
        
        # 4. Meta Ensemble (ترکیب تمام ensemble‌ها)
        print("\n4. ایجاد Meta Ensemble...")
        
        # جمع‌آوری تمام پیش‌بینی‌ها
        all_predictions = pd.DataFrame({
            'weighted': weighted_pred[:len(voting_pred)],
            'voting': voting_pred,
            'stacking': stacking_pred
        })
        
        # آموزش یک meta-model روی نتایج ensemble‌ها
        meta_model = LogisticRegression()
        meta_model.fit(all_predictions.values, y)
        
        # پیش‌بینی نهایی
        final_pred = meta_model.predict_proba(all_predictions.values)[:, 1]
        
        # ارزیابی
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # آستانه‌بندی
        final_pred_binary = (final_pred > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, final_pred_binary),
            'precision': precision_score(y, final_pred_binary),
            'recall': recall_score(y, final_pred_binary),
            'f1': f1_score(y, final_pred_binary)
        }
        
        print(f"\nنتایج Ensemble نهایی برای {symbol}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        
        # ذخیره ensemble models
        ensemble_models = {
            'weighted_ensemble': available_weights,
            'voting_ensemble': voting_clf,
            'stacking_ensemble': stacking_clf,
            'meta_ensemble': meta_model
        }
        
        # ذخیره مدل‌ها
        ensemble_path = self.ensemble_dir / f"{symbol}_ensemble.pkl"
        joblib.dump(ensemble_models, ensemble_path)
        
        # ذخیره پیش‌بینی‌ها
        preds_df = pd.DataFrame({
            'final_prediction': final_pred,
            'final_signal': final_pred_binary,
            'actual': y
        }, index=ml_preds.index)
        
        preds_path = self.ensemble_dir / f"{symbol}_ensemble_predictions.csv"
        preds_df.to_csv(preds_path)
        
        print(f"\nEnsemble برای {symbol} کامل شد و ذخیره شد")
        
        return {
            'models': ensemble_models,
            'metrics': metrics,
            'predictions': preds_df
        }
    
    def train_all_ensembles(self):
        """آموزش ensemble برای تمام جفت‌ارزها"""
        ensemble_results = {}
        
        for pair in settings.FOREX_PAIRS[:2]:
            results = self.train_ensemble(pair)
            if results:
                ensemble_results[pair] = results
        
        return ensemble_results

if __name__ == "__main__":
    trainer = EnsembleTrainer()
    results = trainer.train_all_ensembles()
    print(f"آموزش Ensemble برای {len(results)} جفت کامل شد")