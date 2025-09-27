# src/models/base/lstm_model.py
"""
LSTM model implementation for time series forecasting
"""

import numpy as np
import warnings
import logging
import pickle
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
tf.random.set_seed(42)
np.random.seed(42)


class LSTMModel:
    """
    Advanced LSTM Model implementation for time series forecasting
    """

    def __init__(self,
                 sequence_length: int = 30,
                 lstm_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 batch_norm: bool = False,
                 l2_reg: float = 0.0,
                 scaler_type: str = 'minmax'):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.scaler_type = scaler_type

        self.model = None
        self.scaler = self._create_scaler()
        self.is_trained = False
        self.training_history = None
        self.input_shape = None
        self.best_model_path = None

    def _create_scaler(self):
        if self.scaler_type == 'minmax':
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            return StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")

    def _create_sequences(self, data: np.ndarray, target: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        if target is None:
            target = data
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential()
        model.add(Input(shape=input_shape))
        for i, units in enumerate(self.lstm_units):
            return_seq = i < len(self.lstm_units) - 1
            model.add(LSTM(
                units,
                return_sequences=return_seq,
                kernel_regularizer=l2(self.l2_reg) if self.l2_reg > 0 else None,
                name=f'lstm_{i+1}'
            ))
            if self.batch_norm:
                model.add(BatchNormalization())
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, name='output'))
        opt = Adam(learning_rate=self.learning_rate) if self.optimizer_name.lower() == 'adam' \
              else RMSprop(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return model

    def _prepare_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        arr = data.reshape(-1, 1)
        if fit:
            scaled = self.scaler.fit_transform(arr)
        else:
            scaled = self.scaler.transform(arr)
        return scaled.flatten()

    def _create_callbacks(self, checkpoint_path: Optional[str], patience: int
                          ) -> List[Any]:
        cbs = [
            EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=patience//2,
                              factor=0.5, min_lr=1e-7, verbose=1)
        ]
        if checkpoint_path:
            cbs.append(ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ))
            self.best_model_path = checkpoint_path
        return cbs

    def train(self,
              train_data: np.ndarray,
              validation_data: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              verbose: int = 1,
              model_checkpoint_path: Optional[str] = None,
              early_stopping_patience: int = 15) -> Dict[str, Any]:
        logger.info(f"Training LSTM model with units {self.lstm_units}")
        train_scaled = self._prepare_data(train_data, fit=True)
        X_train, y_train = self._create_sequences(train_scaled)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.input_shape = (X_train.shape[1], 1)

        val_processed = None
        if validation_data is not None:
            val_scaled = self._prepare_data(validation_data, fit=False)
            X_val, y_val = self._create_sequences(val_scaled)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            val_processed = (X_val, y_val)
            validation_split = None

        self.model = self._build_model(self.input_shape)
        if verbose:
            print("Model Architecture:")
            self.model.summary()
        callbacks = self._create_callbacks(model_checkpoint_path, early_stopping_patience)

        history = self.model.fit(
            X_train, y_train,
            validation_data=val_processed,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False
        )
        self.is_trained = True
        self.training_history = history.history

        results = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history.get('val_loss', [None])[-1],
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': min(history.history.get('val_loss', [history.history['loss'][-1]])),
            'early_stopped': len(history.history['loss']) < epochs
        }
        logger.info(f"LSTM training completed. Loss: {results['final_train_loss']:.4f}")
        return results

    def predict(self,
                input_data: np.ndarray,
                steps: int = 1,
                return_sequences: bool = False) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before predictions")
        seq = self._prepare_data(input_data, fit=False)[-self.sequence_length:]
        preds, seqs = [], []
        for _ in range(steps):
            inp = seq.reshape(1, self.sequence_length, 1)
            p = self.model.predict(inp, verbose=0)[0, 0]
            if return_sequences:
                seqs.append(seq.copy())
            seq = np.append(seq[1:], p)
            preds.append(p)
        arr = np.array(preds).reshape(-1, 1)
        inv = self.scaler.inverse_transform(arr).flatten()
        if return_sequences:
            inv_seqs = [self.scaler.inverse_transform(s.reshape(-1, 1)).flatten() for s in seqs]
            return {'predictions': inv, 'sequences': inv_seqs}
        return inv

    def save_model(self, filepath: str):
        """Save trained LSTM model in native Keras format (.keras)"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        # บันทึกเป็น .keras แทน .h5
        keras_path = f"{filepath}.keras"
        self.model.save(keras_path, include_optimizer=False, save_format='keras')
        # Save additional components
        import pickle
        comp = {
            'scaler': self.scaler,
            'params': self.get_params(),
            'history': self.training_history,
            'input_shape': self.input_shape
        }
        with open(f"{filepath}_components.pkl", "wb") as f:
            pickle.dump(comp, f)
        logger.info(f"LSTM model saved to {keras_path}")

    def load_model(self, filepath: str):
        """Load trained model from native Keras format (.keras)"""
        # โหลดจากไฟล์ .keras
        self.model = tf.keras.models.load_model(f"{filepath}.keras", compile=False)
        # โหลด components
        import pickle
        with open(f"{filepath}_components.pkl", "rb") as f:
            comp = pickle.load(f)
        self.scaler = comp['scaler']
        self.training_history = comp['history']
        self.input_shape = comp['input_shape']
        # กำหนดค่าพารามิเตอร์ต่าง ๆ
        params = comp['params']
        self.sequence_length = params['sequence_length']
        self.lstm_units = params['lstm_units']
        self.dropout_rate = params['dropout_rate']
        self.learning_rate = params['learning_rate']
        self.optimizer_name = params['optimizer']
        self.batch_norm = params['batch_norm']
        self.l2_reg = params['l2_reg']
        self.scaler_type = params['scaler_type']
        self.is_trained = True
        logger.info(f"LSTM model loaded from {filepath}.keras")

    def get_params(self) -> Dict[str, Any]:
        return {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer_name,
            'batch_norm': self.batch_norm,
            'l2_reg': self.l2_reg,
            'scaler_type': self.scaler_type,
            'is_trained': self.is_trained
        }

    def evaluate(self, test_data: np.ndarray, true_vals: np.ndarray) -> Dict[str, float]:
        preds = self.predict(test_data, steps=len(true_vals))
        mae = mean_absolute_error(true_vals, preds)
        mse = mean_squared_error(true_vals, preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true_vals - preds) / np.where(true_vals != 0, true_vals, 1))) * 100
        smape = 100 * np.mean(2 * np.abs(preds-true_vals)/(np.abs(true_vals)+np.abs(preds)))
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}