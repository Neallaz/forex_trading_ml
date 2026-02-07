"""
Training Deep Learning Models (LSTM, CNN, Attention)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
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
        
        print(f"DEBUG create_sequences:")
        print(f"  Data shape: {data.shape}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Data type: {data.dtype}")
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            target = data[i+sequence_length, -2]  # target column
            sequences.append(seq)
            targets.append(target)
        
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        print(f"DEBUG create_sequences results:")
        print(f"  Sequences shape: {sequences_array.shape}")
        print(f"  Targets shape: {targets_array.shape}")
        print(f"  Targets dtype: {targets_array.dtype}")
        print(f"  Targets unique values: {np.unique(targets_array)}")
        print(f"  Targets min: {targets_array.min():.6f}, max: {targets_array.max():.6f}")
        print(f"  First 5 targets: {targets_array[:5]}")
        
        return sequences_array, targets_array
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        print(f"DEBUG build_lstm_model: input_shape = {input_shape}")
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
        print(f"DEBUG: Looking for features file at: {features_path}")
        print(f"DEBUG: File exists: {features_path.exists()}")
        
        if not features_path.exists():
            print(f"Features file for {symbol} not found")
            return None
        
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        print(f"\nDEBUG DATA LOADING:")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame dtypes:\n{df.dtypes}")
        
        # Check target columns
        if 'target' not in df.columns:
            print("ERROR: 'target' column not found in DataFrame!")
            return None
            
        if 'target_return' not in df.columns:
            print("WARNING: 'target_return' column not found in DataFrame")
        
        print(f"\nDEBUG TARGET STATISTICS:")
        print(f"Target column 'target' statistics:")
        print(df['target'].describe())
        print(f"\nTarget unique values: {df['target'].unique()[:20]}")  # Show first 20 unique values
        print(f"Number of unique target values: {len(df['target'].unique())}")
        print(f"Target min: {df['target'].min():.6f}, max: {df['target'].max():.6f}")
        print(f"Target sample values (first 10):")
        print(df['target'].head(10).values)
        
        # Check if target is binary
        unique_targets = df['target'].unique()
        if len(unique_targets) <= 2:
            print(f"\nINFO: Target appears to be binary with values: {unique_targets}")
        else:
            print(f"\nWARNING: Target has {len(unique_targets)} unique values - likely continuous")
            print(f"  This might cause issues with binary classification metrics!")
        
        # Separate features and target
        X = df.drop(['target', 'target_return'], axis=1).values
        y = df['target'].values.astype(float)
        
        print(f"\nDEBUG FEATURES AND TARGET:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y dtype: {y.dtype}")
        print(f"y unique values: {np.unique(y)}")
        print(f"y min: {y.min():.6f}, max: {y.max():.6f}")
        
        # Normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"\nDEBUG NORMALIZATION:")
        print(f"X_scaled shape: {X_scaled.shape}")
        print(f"X_scaled mean (first feature): {X_scaled[:, 0].mean():.6f}")
        print(f"X_scaled std (first feature): {X_scaled[:, 0].std():.6f}")
        
        # Create sequences
        sequence_length = settings.FEATURE_WINDOW
        print(f"\nDEBUG SEQUENCE CREATION:")
        print(f"Sequence length: {sequence_length}")
        print(f"Combining X_scaled (shape: {X_scaled.shape}) with y (shape: {y.shape})")
        
        combined_data = np.column_stack([X_scaled, y])
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Last column (target) in combined data sample: {combined_data[:5, -1]}")
        
        X_seq, y_seq = self.create_sequences(
            combined_data, 
            sequence_length
        )
        
        # Split data
        split_idx = int(len(X_seq) * settings.TRAIN_TEST_SPLIT)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"\nDEBUG DATA SPLITTING:")
        print(f"Total sequences: {len(X_seq)}")
        print(f"Train split index: {split_idx}")
        print(f"Train sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"y_test unique values: {np.unique(y_test)}")
        print(f"y_test sample: {y_test[:10]}")
        
        # Build different models
        input_shape = (sequence_length, X_train.shape[2])
        models = {}
        
        # 1. Simple LSTM model
        print("\n1. Training LSTM model...")
        lstm_model = self.build_lstm_model(input_shape)
        lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"\nDEBUG MODEL SUMMARY:")
        lstm_model.summary()
        
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
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
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
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
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
            print(f"\n{'='*30}")
            print(f"Evaluating {model_name} model...")
            print(f"{'='*30}")
            
            # Predict
            y_pred = model.predict(X_test)
            
            print(f"\nDEBUG PREDICTIONS for {model_name}:")
            print(f"y_pred shape: {y_pred.shape}")
            print(f"y_pred dtype: {y_pred.dtype}")
            print(f"y_pred min: {y_pred.min():.6f}, max: {y_pred.max():.6f}")
            print(f"y_pred mean: {y_pred.mean():.6f}")
            print(f"Sample y_pred values (first 10): {y_pred[:10].flatten()}")
            
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            print(f"\nDEBUG BINARY PREDICTIONS for {model_name}:")
            print(f"y_pred_binary shape: {y_pred_binary.shape}")
            print(f"y_pred_binary dtype: {y_pred_binary.dtype}")
            print(f"y_pred_binary unique values: {np.unique(y_pred_binary)}")
            print(f"y_pred_binary distribution: 0s: {(y_pred_binary == 0).sum()}, 1s: {(y_pred_binary == 1).sum()}")
            print(f"Sample y_pred_binary values (first 10): {y_pred_binary[:10].flatten()}")
            
            print(f"\nDEBUG y_test for {model_name}:")
            print(f"y_test shape: {y_test.shape}")
            print(f"y_test dtype: {y_test.dtype}")
            print(f"y_test unique values: {np.unique(y_test)}")
            print(f"y_test min: {y_test.min():.6f}, max: {y_test.max():.6f}")
            print(f"Sample y_test values (first 10): {y_test[:10]}")
            
            # Check if we can calculate classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            print(f"\nDEBUG METRIC CALCULATION for {model_name}:")
            
            try:
                # Try to calculate metrics
                accuracy = accuracy_score(y_test, y_pred_binary)
                print(f"  Accuracy calculated successfully: {accuracy:.4f}")
                
                # Check if we need to handle edge cases for precision/recall
                if len(np.unique(y_pred_binary)) < 2:
                    print(f"  WARNING: y_pred_binary has only {len(np.unique(y_pred_binary))} unique value(s)")
                    precision = 0.0 if (y_pred_binary == 1).sum() == 0 else precision_score(y_test, y_pred_binary, zero_division=0)
                    recall = 0.0 if (y_pred_binary == 1).sum() == 0 else recall_score(y_test, y_pred_binary, zero_division=0)
                    f1 = 0.0 if (y_pred_binary == 1).sum() == 0 else f1_score(y_test, y_pred_binary, zero_division=0)
                else:
                    precision = precision_score(y_test, y_pred_binary, zero_division=0)
                    recall = recall_score(y_test, y_pred_binary, zero_division=0)
                    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'model': model_name
                }
                
            except ValueError as e:
                print(f"  ERROR calculating metrics: {e}")
                print(f"  Attempting to convert y_test to binary for compatibility...")
                
                # Convert y_test to binary if it's continuous
                y_test_binary = (y_test > 0).astype(int)
                print(f"  Converted y_test to binary with values: {np.unique(y_test_binary)}")
                
                metrics = {
                    'accuracy': accuracy_score(y_test_binary, y_pred_binary),
                    'precision': precision_score(y_test_binary, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_test_binary, y_pred_binary, zero_division=0),
                    'f1': f1_score(y_test_binary, y_pred_binary, zero_division=0),
                    'model': model_name
                }
            
            results[model_name] = metrics
            predictions[model_name] = y_pred.flatten()
            
            print(f"\nResults for {model_name}:")
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
    print("Starting DL Model Training with Debug Information...")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Processed Data Dir: {settings.PROCESSED_DATA_DIR}")
    print(f"Models Dir: {settings.MODELS_DIR}")
    
    trainer = DLModelTrainer()
    results = trainer.train_for_all_pairs()
    print(f"\n{'='*50}")
    print(f"DL model training completed for {len(results)} pairs")
    print(f"{'='*50}")
    
# """
# Training Deep Learning Models (LSTM, CNN, Attention)
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')
# import os
# import sys
# # Add project root to Python path
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, PROJECT_ROOT)
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import (
#     LSTM, GRU, Dense, Dropout, Bidirectional,
#     Conv1D, MaxPooling1D, Flatten,
#     Input, Attention, GlobalAveragePooling1D,
#     BatchNormalization, LayerNormalization
# )
# from tensorflow.keras.callbacks import (
#     EarlyStopping, ReduceLROnPlateau,
#     ModelCheckpoint, TensorBoard
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2

# from sklearn.preprocessing import StandardScaler
# import joblib

# from config.settings import settings

# class DLModelTrainer:
#     """Deep Learning Model Trainer Class"""
    
#     def __init__(self):
#         self.processed_dir = Path(settings.PROCESSED_DATA_DIR)
#         self.models_dir = Path(settings.MODELS_DIR) / "dl"
#         self.models_dir.mkdir(parents=True, exist_ok=True)
        
#     def create_sequences(self, data, sequence_length=60):
#         """Create sequences for time-based models"""
#         sequences = []
#         targets = []
        
#         for i in range(len(data) - sequence_length):
#             seq = data[i:i+sequence_length]
#             target = data[i+sequence_length, -2]  # target column
#             sequences.append(seq)
#             targets.append(target)
        
#         return np.array(sequences), np.array(targets)
    
#     def build_lstm_model(self, input_shape):
#         """Build LSTM model"""
#         model = Sequential([
#             Input(shape=input_shape),
#             LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
#             Dropout(0.3),
#             BatchNormalization(),
            
#             LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
#             Dropout(0.3),
#             BatchNormalization(),
            
#             LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
#             Dropout(0.2),
#             BatchNormalization(),
            
#             Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
#             Dropout(0.2),
#             Dense(1, activation='sigmoid')
#         ])
        
#         return model
    
#     def build_bidirectional_lstm(self, input_shape):
#         """Build Bidirectional LSTM model"""
#         model = Sequential([
#             Input(shape=input_shape),
#             Bidirectional(LSTM(64, return_sequences=True)),
#             Dropout(0.3),
#             BatchNormalization(),
            
#             Bidirectional(LSTM(32, return_sequences=False)),
#             Dropout(0.2),
#             BatchNormalization(),
            
#             Dense(16, activation='relu'),
#             Dropout(0.1),
#             Dense(1, activation='sigmoid')
#         ])
        
#         return model
    
#     def build_lstm_with_attention(self, input_shape):
#         """Build LSTM model with Attention Mechanism"""
#         inputs = Input(shape=input_shape)
        
#         # LSTM layers
#         lstm1 = LSTM(64, return_sequences=True)(inputs)
#         dropout1 = Dropout(0.3)(lstm1)
#         bn1 = BatchNormalization()(dropout1)
        
#         lstm2 = LSTM(32, return_sequences=True)(bn1)
#         dropout2 = Dropout(0.2)(lstm2)
#         bn2 = BatchNormalization()(dropout2)
        
#         # Attention mechanism
#         attention = Attention()([bn2, bn2])
#         attention_pool = GlobalAveragePooling1D()(attention)
        
#         # Dense layers
#         dense1 = Dense(32, activation='relu')(attention_pool)
#         dropout3 = Dropout(0.2)(dense1)
#         outputs = Dense(1, activation='sigmoid')(dropout3)
        
#         model = Model(inputs=inputs, outputs=outputs)
#         return model
    
#     def build_cnn_lstm_hybrid(self, input_shape):
#         """Build CNN-LSTM hybrid model"""
#         model = Sequential([
#             Input(shape=input_shape),
            
#             # CNN layers for feature extraction
#             Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
#             BatchNormalization(),
#             MaxPooling1D(pool_size=2),
            
#             Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
#             BatchNormalization(),
#             MaxPooling1D(pool_size=2),
            
#             # LSTM layers for temporal patterns
#             LSTM(64, return_sequences=True),
#             Dropout(0.3),
#             BatchNormalization(),
            
#             LSTM(32, return_sequences=False),
#             Dropout(0.2),
#             BatchNormalization(),
            
#             Dense(16, activation='relu'),
#             Dropout(0.1),
#             Dense(1, activation='sigmoid')
#         ])
        
#         return model
    
#     def create_custom_loss(self):
#         """Create custom loss function considering financial metrics"""
#         def sharpe_loss(y_true, y_pred):
#             # Calculate returns based on prediction
#             returns = y_pred * y_true
            
#             # Calculate Sharpe Ratio
#             mean_return = keras.backend.mean(returns)
#             std_return = keras.backend.std(returns)
#             sharpe_ratio = mean_return / (std_return + keras.backend.epsilon())
            
#             # Minimize negative sharpe (maximize sharpe)
#             return -sharpe_ratio
        
#         return sharpe_loss
    
#     def train_dl_model(self, symbol):
#         """Train Deep Learning model for a currency pair"""
#         print(f"\n{'='*50}")
#         print(f"Training DL models for {symbol}")
#         print(f"{'='*50}")
        
#         # Load data
#         features_path = self.processed_dir / f"{symbol}_features.csv"
#         if not features_path.exists():
#             print(f"Features file for {symbol} not found")
#             return None
        
#         df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
#         # Separate features and target
#         X = df.drop(['target', 'target_return'], axis=1).values
#         y = df['target'].values.astype(float)
        
#         # Normalization
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Create sequences
#         sequence_length = settings.FEATURE_WINDOW
#         X_seq, y_seq = self.create_sequences(
#             np.column_stack([X_scaled, y]), 
#             sequence_length
#         )
        
#         # Split data
#         split_idx = int(len(X_seq) * settings.TRAIN_TEST_SPLIT)
#         X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
#         y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
#         print(f"Number of sequences - Train: {len(X_train)}, Test: {len(X_test)}")
#         print(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")
        
#         # Build different models
#         input_shape = (sequence_length, X_train.shape[2])
#         models = {}
        
#         # 1. Simple LSTM model
#         print("\n1. Training LSTM model...")
#         lstm_model = self.build_lstm_model(input_shape)
#         lstm_model.compile(
#             optimizer=Adam(learning_rate=0.001),
#             loss='binary_crossentropy',
#             metrics=['accuracy', tf.keras.metrics.AUC()]
#         )
        
#         lstm_history = lstm_model.fit(
#             X_train, y_train,
#             validation_split=0.2,
#             epochs=50,
#             batch_size=32,
#             callbacks=[
#                 EarlyStopping(patience=10, restore_best_weights=True),
#                 ReduceLROnPlateau(factor=0.5, patience=5)
#             ],
#             verbose=1
#         )
        
#         models['lstm'] = lstm_model
        
#         # 2. LSTM with Attention model
#         print("\n2. Training LSTM with Attention model...")
#         attention_model = self.build_lstm_with_attention(input_shape)
#         attention_model.compile(
#             optimizer=Adam(learning_rate=0.0005),
#             loss='binary_crossentropy',
#             metrics=['accuracy', tf.keras.metrics.AUC()]
#         )
        
#         attention_history = attention_model.fit(
#             X_train, y_train,
#             validation_split=0.2,
#             epochs=50,
#             batch_size=32,
#             callbacks=[
#                 EarlyStopping(patience=10, restore_best_weights=True),
#                 ReduceLROnPlateau(factor=0.5, patience=5),
#                 ModelCheckpoint(
#                     str(self.models_dir / f"{symbol}_attention_best.h5"),
#                     save_best_only=True
#                 )
#             ],
#             verbose=1
#         )
        
#         models['attention'] = attention_model
        
#         # 3. CNN-LSTM Hybrid model
#         print("\n3. Training CNN-LSTM Hybrid model...")
#         cnn_lstm_model = self.build_cnn_lstm_hybrid(input_shape)
#         cnn_lstm_model.compile(
#             optimizer=Adam(learning_rate=0.001),
#             loss='binary_crossentropy',
#             metrics=['accuracy', tf.keras.metrics.AUC()]
#         )
        
#         cnn_lstm_history = cnn_lstm_model.fit(
#             X_train, y_train,
#             validation_split=0.2,
#             epochs=50,
#             batch_size=32,
#             callbacks=[
#                 EarlyStopping(patience=10, restore_best_weights=True),
#                 ReduceLROnPlateau(factor=0.5, patience=5)
#             ],
#             verbose=1
#         )
        
#         models['cnn_lstm'] = cnn_lstm_model
        
#         # Evaluate models
#         results = {}
#         predictions = {}
        
#         for model_name, model in models.items():
#             print(f"\nEvaluating {model_name} model...")
            
#             # Predict
#             y_pred = model.predict(X_test)
#             y_pred_binary = (y_pred > 0.5).astype(int)
            
#             # Calculate metrics
#             from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
#             metrics = {
#                 'accuracy': accuracy_score(y_test, y_pred_binary),
#                 'precision': precision_score(y_test, y_pred_binary),
#                 'recall': recall_score(y_test, y_pred_binary),
#                 'f1': f1_score(y_test, y_pred_binary),
#                 'model': model_name
#             }
            
#             results[model_name] = metrics
#             predictions[model_name] = y_pred.flatten()
            
#             print(f"Accuracy: {metrics['accuracy']:.4f}")
#             print(f"Precision: {metrics['precision']:.4f}")
#             print(f"Recall: {metrics['recall']:.4f}")
#             print(f"F1-Score: {metrics['f1']:.4f}")
        
#         # Save models
#         for model_name, model in models.items():
#             model_path = self.models_dir / f"{symbol}_{model_name}.h5"
#             model.save(model_path)
#             print(f"Model {model_name} saved to {model_path}")
        
#         # Save results
#         results_df = pd.DataFrame(results.values())
#         results_path = self.models_dir / f"{symbol}_dl_results.csv"
#         results_df.to_csv(results_path)
        
#         # Save predictions
#         preds_df = pd.DataFrame(predictions)
#         preds_df['actual'] = y_test
#         preds_path = self.models_dir / f"{symbol}_dl_predictions.csv"
#         preds_df.to_csv(preds_path)
        
#         # Save scaler
#         scaler_path = self.models_dir / f"{symbol}_scaler.pkl"
#         joblib.dump(scaler, scaler_path)
        
#         print(f"\nDL model training for {symbol} completed")
        
#         return {
#             'models': models,
#             'results': results_df,
#             'predictions': preds_df,
#             'X_test': X_test,
#             'y_test': y_test
#         }
    
#     def train_for_all_pairs(self):
#         """Train DL models for all currency pairs"""
#         dl_results = {}
        
#         for pair in settings.FOREX_PAIRS[:2]:  # Start with two pairs
#             results = self.train_dl_model(pair)
#             if results:
#                 dl_results[pair] = results
        
#         return dl_results

# if __name__ == "__main__":
#     trainer = DLModelTrainer()
#     results = trainer.train_for_all_pairs()
#     print(f"DL model training completed for {len(results)} pairs")