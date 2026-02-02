"""
آموزش مدل‌های Deep Learning (LSTM, CNN, Attention)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten,
    Input, Attention, GlobalAveragePooling1D,
    BatchNormalization, LayerNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
import joblib

from config.settings import settings

class DLModelTrainer:
    """کلاس آموزش مدل‌های Deep Learning"""
    
    def __init__(self):
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.models_dir = Path(settings.MODELS_DIR) / "dl"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sequences(self, data, sequence_length=60):
        """ایجاد sequences برای مدل‌های مبتنی بر زمان"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            target = data[i+sequence_length, -2]  # target column
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """ساخت مدل LSTM"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_bidirectional_lstm(self, input_shape):
        """ساخت مدل Bidirectional LSTM"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),
            
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_lstm_with_attention(self, input_shape):
        """ساخت مدل LSTM با Attention Mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = LSTM(64, return_sequences=True)(inputs)
        dropout1 = Dropout(0.3)(lstm1)
        bn1 = BatchNormalization()(dropout1)
        
        lstm2 = LSTM(32, return_sequences=True)(bn1)
        dropout2 = Dropout(0.2)(lstm2)
        bn2 = BatchNormalization()(dropout2)
        
        # Attention mechanism
        attention = Attention()([bn2, bn2])
        attention_pool = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(32, activation='relu')(attention_pool)
        dropout3 = Dropout(0.2)(dense1)
        outputs = Dense(1, activation='sigmoid')(dropout3)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_lstm_hybrid(self, input_shape):
        """ساخت مدل ترکیبی CNN-LSTM"""
        model = Sequential([
            Input(shape=input_shape),
            
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # LSTM layers for temporal patterns
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def create_custom_loss(self):
        """ایجاد loss function سفارشی با توجه به financial metrics"""
        def sharpe_loss(y_true, y_pred):
            # محاسبه returns بر اساس پیش‌بینی
            returns = y_pred * y_true
            
            # محاسبه Sharpe Ratio
            mean_return = keras.backend.mean(returns)
            std_return = keras.backend.std(returns)
            sharpe_ratio = mean_return / (std_return + keras.backend.epsilon())
            
            # minimize negative sharpe (maximize sharpe)
            return -sharpe_ratio
        
        return sharpe_loss
    
    def train_dl_model(self, symbol):
        """آموزش مدل Deep Learning برای یک جفت ارز"""
        print(f"\n{'='*50}")
        print(f"آموزش مدل‌های DL برای {symbol}")
        print(f"{'='*50}")
        
        # بارگذاری داده‌ها
        features_path = self.processed_dir / f"{symbol}_features.csv"
        if not features_path.exists():
            print(f"فایل features برای {symbol} یافت نشد")
            return None
        
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        # جدا کردن features و target
        X = df.drop(['target', 'target_return'], axis=1).values
        y = df['target'].values.astype(float)
        
        # Normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ایجاد sequences
        sequence_length = settings.FEATURE_WINDOW
        X_seq, y_seq = self.create_sequences(
            np.column_stack([X_scaled, y]), 
            sequence_length
        )
        
        # تقسیم داده‌ها
        split_idx = int(len(X_seq) * settings.TRAIN_TEST_SPLIT)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"تعداد sequences - Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        # ساخت مدل‌های مختلف
        input_shape = (sequence_length, X_train.shape[2])
        models = {}
        
        # 1. مدل LSTM ساده
        print("\n1. آموزش مدل LSTM...")
        lstm_model = self.build_lstm_model(input_shape)
        lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        models['lstm'] = lstm_model
        
        # 2. مدل LSTM با Attention
        print("\n2. آموزش مدل LSTM with Attention...")
        attention_model = self.build_lstm_with_attention(input_shape)
        attention_model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        attention_history = attention_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
                ModelCheckpoint(
                    str(self.models_dir / f"{symbol}_attention_best.h5"),
                    save_best_only=True
                )
            ],
            verbose=1
        )
        
        models['attention'] = attention_model
        
        # 3. مدل ترکیبی CNN-LSTM
        print("\n3. آموزش مدل CNN-LSTM Hybrid...")
        cnn_lstm_model = self.build_cnn_lstm_hybrid(input_shape)
        cnn_lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        cnn_lstm_history = cnn_lstm_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        models['cnn_lstm'] = cnn_lstm_model
        
        # ارزیابی مدل‌ها
        results = {}
        predictions = {}
        
        for model_name, model in models.items():
            print(f"\nارزیابی مدل {model_name}...")
            
            # پیش‌بینی
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # محاسبه metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred_binary),
                'precision': precision_score(y_test, y_pred_binary),
                'recall': recall_score(y_test, y_pred_binary),
                'f1': f1_score(y_test, y_pred_binary),
                'model': model_name
            }
            
            results[model_name] = metrics
            predictions[model_name] = y_pred.flatten()
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
        
        # ذخیره مدل‌ها
        for model_name, model in models.items():
            model_path = self.models_dir / f"{symbol}_{model_name}.h5"
            model.save(model_path)
            print(f"مدل {model_name} در {model_path} ذخیره شد")
        
        # ذخیره results
        results_df = pd.DataFrame(results.values())
        results_path = self.models_dir / f"{symbol}_dl_results.csv"
        results_df.to_csv(results_path)
        
        # ذخیره predictions
        preds_df = pd.DataFrame(predictions)
        preds_df['actual'] = y_test
        preds_path = self.models_dir / f"{symbol}_dl_predictions.csv"
        preds_df.to_csv(preds_path)
        
        # ذخیره scaler
        scaler_path = self.models_dir / f"{symbol}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        print(f"\nآموزش مدل‌های DL برای {symbol} کامل شد")
        
        return {
            'models': models,
            'results': results_df,
            'predictions': preds_df,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def train_for_all_pairs(self):
        """آموزش مدل‌های DL برای تمام جفت‌ارزها"""
        dl_results = {}
        
        for pair in settings.FOREX_PAIRS[:2]:  # برای شروع دو جفت
            results = self.train_dl_model(pair)
            if results:
                dl_results[pair] = results
        
        return dl_results

if __name__ == "__main__":
    trainer = DLModelTrainer()
    results = trainer.train_for_all_pairs()
    print(f"آموزش مدل‌های DL برای {len(results)} جفت کامل شد")