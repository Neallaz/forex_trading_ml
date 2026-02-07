# models/dl/train_dl.py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

from config.settings import model_config, path_config
from utils.metrics import TradingMetrics
from loguru import logger

class DLModelTrainer:
    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.models_dir = Path(path_config.DL_MODELS_DIR)
        self.data_path = Path(path_config.PROCESSED_DATA_DIR) / f"{symbol.replace('/', '_')}_features.csv"
        
        # تنظیمات TensorFlow
        tf.random.set_seed(42)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
    def load_and_prepare_data(self):
        """بارگذاری و آماده‌سازی داده‌ها برای DL"""
        # بارگذاری داده‌ها
        df = pd.read_csv(self.data_path, index_col='timestamp', parse_dates=True)
        
        # جدا کردن features و labels
        feature_cols = [col for col in df.columns if not col.startswith('label_')]
        X = df[feature_cols].values
        y = df['label_direction'].values
        
        # ساخت دنباله‌های زمانی
        lookback = model_config.LOOKBACK_WINDOW
        X_seq, y_seq = self.create_sequences(X, y, lookback)
        
        # تقسیم داده
        split_idx = int(len(X_seq) * model_config.TRAIN_TEST_SPLIT)
        
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        
        # Standardization
        self.scaler = StandardScaler()
        X_train_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train_shape[-1])
        X_train_scaled_flat = self.scaler.fit_transform(X_train_flat)
        X_train = X_train_scaled_flat.reshape(X_train_shape)
        
        X_test_shape = X_test.shape
        X_test_flat = X_test.reshape(-1, X_test_shape[-1])
        X_test_scaled_flat = self.scaler.transform(X_test_flat)
        X_test = X_test_scaled_flat.reshape(X_test_shape)
        
        # ذخیره scaler
        scaler_path = self.models_dir / f"{self.symbol}_dl_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Data prepared: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def create_sequences(self, X, y, lookback):
        """ساخت دنباله‌های زمانی"""
        X_seq, y_seq = [], []
        
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """ساخت مدل LSTM"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape,
                       kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def build_attention_lstm_model(self, input_shape):
        """ساخت مدل LSTM با Attention"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = layers.LSTM(128, return_sequences=True)(inputs)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = layers.multiply([lstm_out, attention])
        sent_representation = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(sent_representation)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(sent_representation)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """ساخت مدل ترکیبی CNN-LSTM"""
        model = models.Sequential([
            # CNN layers برای استخراج ویژگی‌های محلی
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                         input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # LSTM layers برای الگوهای زمانی
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """آموزش مدل با callbacks"""
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=model_config.LSTM_PARAMS['patience'],
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.models_dir / f"{self.symbol}_{model_name}_best.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # آموزش مدل
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=model_config.LSTM_PARAMS['epochs'],
            batch_size=model_config.LSTM_PARAMS['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """ارزیابی مدل"""
        # پیش‌بینی
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # محاسبه معیارها
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # گزارش کامل
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics.update({
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        })
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def train_all_dl_models(self):
        """آموزش تمام مدل‌های Deep Learning"""
        logger.info(f"Starting DL model training for {self.symbol}...")
        
        # 1. آماده‌سازی داده‌ها
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        
        # تقسیم validation set
        val_split = int(len(X_train) * (1 - model_config.VALIDATION_SPLIT))
        X_train_fit, X_val = X_train[:val_split], X_train[val_split:]
        y_train_fit, y_val = y_train[:val_split], y_train[val_split:]
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        results = {}
        
        # 2. آموزش مدل LSTM
        logger.info("Training LSTM model...")
        lstm_model = self.build_lstm_model(input_shape)
        lstm_model, lstm_history = self.train_model(
            lstm_model, X_train_fit, y_train_fit, X_val, y_val, "lstm"
        )
        lstm_metrics, lstm_pred, lstm_proba = self.evaluate_model(
            lstm_model, X_test, y_test, "LSTM"
        )
        
        # ذخیره مدل LSTM
        lstm_model.save(self.models_dir / f"{self.symbol}_lstm_model.keras")
        
        results['lstm'] = {
            'model': lstm_model,
            'metrics': lstm_metrics,
            'history': lstm_history.history,
            'predictions': lstm_pred,
            'probabilities': lstm_proba
        }
        
        # 3. آموزش مدل Attention LSTM
        logger.info("Training Attention LSTM model...")
        attention_model = self.build_attention_lstm_model(input_shape)
        attention_model, attention_history = self.train_model(
            attention_model, X_train_fit, y_train_fit, X_val, y_val, "attention"
        )
        attention_metrics, attention_pred, attention_proba = self.evaluate_model(
            attention_model, X_test, y_test, "Attention LSTM"
        )
        
        # ذخیره مدل Attention
        attention_model.save(self.models_dir / f"{self.symbol}_attention_model.keras")
        
        results['attention'] = {
            'model': attention_model,
            'metrics': attention_metrics,
            'history': attention_history.history,
            'predictions': attention_pred,
            'probabilities': attention_proba
        }
        
        # 4. آموزش مدل CNN-LSTM
        logger.info("Training CNN-LSTM model...")
        cnn_lstm_model = self.build_cnn_lstm_model(input_shape)
        cnn_lstm_model, cnn_lstm_history = self.train_model(
            cnn_lstm_model, X_train_fit, y_train_fit, X_val, y_val, "cnn_lstm"
        )
        cnn_lstm_metrics, cnn_lstm_pred, cnn_lstm_proba = self.evaluate_model(
            cnn_lstm_model, X_test, y_test, "CNN-LSTM"
        )
        
        # ذخیره مدل CNN-LSTM
        cnn_lstm_model.save(self.models_dir / f"{self.symbol}_cnn_lstm_model.keras")
        
        results['cnn_lstm'] = {
            'model': cnn_lstm_model,
            'metrics': cnn_lstm_metrics,
            'history': cnn_lstm_history.history,
            'predictions': cnn_lstm_pred,
            'probabilities': cnn_lstm_proba
        }
        
        # 5. ذخیره نتایج
        self.save_dl_results(results)
        
        logger.info("DL model training completed!")
        return results
    
    def save_dl_results(self, results):
        """ذخیره نتایج DL"""
        # ذخیره metrics
        metrics_list = []
        for model_name, result in results.items():
            metrics = result['metrics'].copy()
            metrics['model'] = model_name
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        metrics_path = self.models_dir / f"{self.symbol}_dl_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"DL metrics saved to {metrics_path}")
        
        # ذخیره predictions
        for model_name, result in results.items():
            pred_df = pd.DataFrame({
                'actual': result.get('actual', []),
                'predicted': result['predictions'],
                'probability': result['probabilities']
            })
            pred_path = self.models_dir / f"{self.symbol}_{model_name}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)
            
            logger.info(f"{model_name} predictions saved to {pred_path}")

def train_all_dl_models():
    """تابع اصلی برای آموزش مدل‌های DL"""
    trainer = DLModelTrainer('EURUSD')
    results = trainer.train_all_dl_models()
    return results

if __name__ == "__main__":
    train_all_dl_models()