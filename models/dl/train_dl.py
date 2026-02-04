
"""
Training Deep Learning Models (LSTM, CNN, Attention)
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
    """Deep Learning Model Trainer Class"""
    
    def __init__(self):
        self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
        self.models_dir = Path(settings.MODELS_DIR) / "dl"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sequences(self, data, sequence_length=60):
        """Create sequences for time-based models"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            target = data[i+sequence_length, -2]  # target column
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
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
        """Build Bidirectional LSTM model"""
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
        """Build LSTM model with Attention Mechanism"""
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
        """Build CNN-LSTM hybrid model"""
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
        """Create custom loss function considering financial metrics"""
        def sharpe_loss(y_true, y_pred):
            # Calculate returns based on prediction
            returns = y_pred * y_true
            
            # Calculate Sharpe Ratio
            mean_return = keras.backend.mean(returns)
            std_return = keras.backend.std(returns)
            sharpe_ratio = mean_return / (std_return + keras.backend.epsilon())
            
            # Minimize negative sharpe (maximize sharpe)
            return -sharpe_ratio
        
        return sharpe_loss
    
    def train_dl_model(self, symbol):
        """Train Deep Learning model for a currency pair"""
        print(f"\n{'='*50}")
        print(f"Training DL models for {symbol}")
        print(f"{'='*50}")
        
        # Load data
        features_path = self.processed_dir / f"{symbol}_features.csv"
        if not features_path.exists():
            print(f"Features file for {symbol} not found")
            return None
        
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        # Separate features and target
        X = df.drop(['target', 'target_return'], axis=1).values
        y = df['target'].values.astype(float)
        
        # Normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        sequence_length = settings.FEATURE_WINDOW
        X_seq, y_seq = self.create_sequences(
            np.column_stack([X_scaled, y]), 
            sequence_length
        )
        
        # Split data
        split_idx = int(len(X_seq) * settings.TRAIN_TEST_SPLIT)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"Number of sequences - Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        # Build different models
        input_shape = (sequence_length, X_train.shape[2])
        models = {}
        
        # 1. Simple LSTM model
        print("\n1. Training LSTM model...")
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
        
        # 2. LSTM with Attention model
        print("\n2. Training LSTM with Attention model...")
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
        
        # 3. CNN-LSTM Hybrid model
        print("\n3. Training CNN-LSTM Hybrid model...")
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
        
        # Evaluate models
        results = {}
        predictions = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
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
        
        # Save models
        for model_name, model in models.items():
            model_path = self.models_dir / f"{symbol}_{model_name}.h5"
            model.save(model_path)
            print(f"Model {model_name} saved to {model_path}")
        
        # Save results
        results_df = pd.DataFrame(results.values())
        results_path = self.models_dir / f"{symbol}_dl_results.csv"
        results_df.to_csv(results_path)
        
        # Save predictions
        preds_df = pd.DataFrame(predictions)
        preds_df['actual'] = y_test
        preds_path = self.models_dir / f"{symbol}_dl_predictions.csv"
        preds_df.to_csv(preds_path)
        
        # Save scaler
        scaler_path = self.models_dir / f"{symbol}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        print(f"\nDL model training for {symbol} completed")
        
        return {
            'models': models,
            'results': results_df,
            'predictions': preds_df,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def train_for_all_pairs(self):
        """Train DL models for all currency pairs"""
        dl_results = {}
        
        for pair in settings.FOREX_PAIRS[:2]:  # Start with two pairs
            results = self.train_dl_model(pair)
            if results:
                dl_results[pair] = results
        
        return dl_results

if __name__ == "__main__":
    trainer = DLModelTrainer()
    results = trainer.train_for_all_pairs()
    print(f"DL model training completed for {len(results)} pairs")