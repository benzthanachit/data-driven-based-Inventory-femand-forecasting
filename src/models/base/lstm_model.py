# Individual Model Implementations - LSTM Model

"""
src/models/base/lstm_model.py
LSTM model implementation for time series forecasting
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
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
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Input sequence length
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'rmsprop')
            batch_norm: Whether to use batch normalization
            l2_reg: L2 regularization strength
            scaler_type: Type of scaler ('minmax', 'standard')
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.scaler_type = scaler_type
        
        # Model components
        self.model = None
        self.scaler = self._create_scaler()
        self.is_trained = False
        self.training_history = None
        self.input_shape = None
        
        # Training parameters
        self.best_model_path = None
        
    def _create_scaler(self):
        """Create appropriate scaler"""
        if self.scaler_type == 'minmax':
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            return StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    def _create_sequences(self, data: np.ndarray, target_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training/prediction
        
        Args:
            data: Input data
            target_data: Target data (if different from input)
            
        Returns:
            X, y sequences
        """
        if target_data is None:
            target_data = data
            
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple, output_dim: int = 1) -> Sequential:
        """
        Build LSTM architecture
        
        Args:
            input_shape: Input shape (sequence_length, features)
            output_dim: Output dimension
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=input_shape))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(LSTM(
                units, 
                return_sequences=return_sequences,
                kernel_regularizer=l2(self.l2_reg) if self.l2_reg > 0 else None,
                name=f'lstm_{i+1}'
            ))
            
            # Batch normalization
            if self.batch_norm:
                model.add(BatchNormalization())
            
            # Dropout
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(output_dim, name='output'))
        
        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _get_optimizer(self):
        """Get optimizer instance"""
        if self.optimizer_name.lower() == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError("Optimizer must be 'adam' or 'rmsprop'")
    
    def _prepare_data(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """
        Prepare and scale data
        
        Args:
            data: Raw data
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Scaled data
        """
        # Reshape for scaler
        data_reshaped = data.reshape(-1, 1)
        
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data_reshaped)
        else:
            scaled_data = self.scaler.transform(data_reshaped)
        
        return scaled_data.flatten()
    
    def _create_callbacks(self, 
                         model_checkpoint_path: Optional[str] = None,
                         patience: int = 15,
                         reduce_lr_patience: int = 7) -> List:
        """
        Create training callbacks
        
        Args:
            model_checkpoint_path: Path to save best model
            patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if model_checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
            self.best_model_path = model_checkpoint_path
        
        return callbacks
    
    def train(self, 
              train_data: np.ndarray,
              validation_data: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              verbose: int = 1,
              model_checkpoint_path: Optional[str] = None,
              early_stopping_patience: int = 15) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Verbosity level
            model_checkpoint_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training LSTM model with {len(self.lstm_units)} layers")
        
        # Prepare training data
        train_scaled = self._prepare_data(train_data, fit_scaler=True)
        X_train, y_train = self._create_sequences(train_scaled)
        
        # Reshape for LSTM input (samples, time steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Prepare validation data
        validation_data_processed = None
        if validation_data is not None:
            val_scaled = self._prepare_data(validation_data, fit_scaler=False)
            X_val, y_val = self._create_sequences(val_scaled)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data_processed = (X_val, y_val)
            validation_split = None
        
        # Build model
        self.model = self._build_model(self.input_shape)
        
        if verbose >= 1:
            print("Model Architecture:")
            self.model.summary()
        
        # Create callbacks
        callbacks = self._create_callbacks(
            model_checkpoint_path=model_checkpoint_path,
            patience=early_stopping_patience
        )
        
        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data_processed,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=False  # Important for time series
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            # Calculate training results
            train_loss = history.history['loss'][-1]
            val_loss = history.history.get('val_loss', [None])[-1]
            epochs_trained = len(history.history['loss'])
            
            results = {
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'epochs_trained': epochs_trained,
                'best_val_loss': min(history.history.get('val_loss', [train_loss])),
                'training_stopped_early': epochs_trained < epochs
            }
            
            logger.info(f"LSTM training completed. Final loss: {train_loss:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            raise
    
    def predict(self, 
               input_data: np.ndarray, 
               steps: int = 1,
               return_sequences: bool = False) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            input_data: Input data for prediction
            steps: Number of steps to predict
            return_sequences: Whether to return intermediate sequences
            
        Returns:
            Predictions (inverse transformed)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale input data
        input_scaled = self._prepare_data(input_data, fit_scaler=False)
        
        # Multi-step prediction
        predictions = []
        current_sequence = input_scaled[-self.sequence_length:]
        sequences_used = []
        
        for step in range(steps):
            # Prepare input for model
            current_input = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            pred_scaled = self.model.predict(current_input, verbose=0)[0, 0]
            
            # Store sequence if requested
            if return_sequences:
                sequences_used.append(current_sequence.copy())
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred_scaled)
            predictions.append(pred_scaled)
        
        # Convert to numpy array and reshape for inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions_original = self.scaler.inverse_transform(predictions).flatten()
        
        if return_sequences:
            # Also inverse transform sequences
            sequences_original = []
            for seq in sequences_used:
                seq_reshaped = seq.reshape(-1, 1)
                seq_original = self.scaler.inverse_transform(seq_reshaped).flatten()
                sequences_original.append(seq_original)
            
            return {
                'predictions': predictions_original,
                'input_sequences': sequences_original
            }
        
        return predictions_original
    
    def predict_with_confidence(self, 
                               input_data: np.ndarray, 
                               steps: int = 1,
                               n_simulations: int = 100,
                               confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Generate predictions with confidence intervals using Monte Carlo simulation
        
        Args:
            input_data: Input data
            steps: Number of steps to predict
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        all_predictions = []
        
        for _ in range(n_simulations):
            # Add noise to input (Monte Carlo simulation)
            noise_std = np.std(input_data) * 0.01  # Small noise
            noisy_input = input_data + np.random.normal(0, noise_std, len(input_data))
            
            # Generate prediction
            pred = self.predict(noisy_input, steps=steps)
            all_predictions.append(pred)
        
        # Calculate statistics
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_ci = np.percentile(all_predictions, upper_percentile, axis=0)
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'all_predictions': all_predictions
        }
    
    def evaluate(self, 
                test_data: np.ndarray, 
                true_values: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            test_data: Test input data
            true_values: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Generate predictions
        predictions = self.predict(test_data, steps=len(true_values))
        
        # Calculate metrics
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE
        mape = np.mean(np.abs((true_values - predictions) / np.where(true_values != 0, true_values, 1))) * 100
        
        # SMAPE
        smape = 100 * np.mean(2 * np.abs(predictions - true_values) / (np.abs(true_values) + np.abs(predictions)))
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape
        }
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """Plot training history"""
        if not self.is_trained or self.training_history is None:
            raise ValueError("Model must be trained before plotting history")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot loss
            ax1.plot(self.training_history['loss'], label='Training Loss')
            if 'val_loss' in self.training_history:
                ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot MAE
            if 'mae' in self.training_history:
                ax2.plot(self.training_history['mae'], label='Training MAE')
            if 'val_mae' in self.training_history:
                ax2.plot(self.training_history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
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
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save the Keras model
        self.model.save(f"{filepath}.h5")
        
        # Save additional components
        import pickle
        model_data = {
            'scaler': self.scaler,
            'params': self.get_params(),
            'training_history': self.training_history,
            'input_shape': self.input_shape
        }
        
        with open(f"{filepath}_components.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        # Load Keras model
        self.model = tf.keras.models.load_model(f"{filepath}.h5")
        
        # Load additional components
        import pickle
        with open(f"{filepath}_components.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.training_history = model_data['training_history']
        self.input_shape = model_data['input_shape']
        
        # Restore parameters
        params = model_data['params']
        self.sequence_length = params['sequence_length']
        self.lstm_units = params['lstm_units']
        self.dropout_rate = params['dropout_rate']
        self.learning_rate = params['learning_rate']
        self.optimizer_name = params['optimizer']
        self.batch_norm = params['batch_norm']
        self.l2_reg = params['l2_reg']
        self.scaler_type = params['scaler_type']
        self.is_trained = params['is_trained']
        
        logger.info(f"LSTM model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    t = np.arange(n_samples)
    data = 50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 2, n_samples)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Initialize model
    model = LSTMModel(
        sequence_length=30,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Train model
    results = model.train(
        train_data,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    print("Training Results:", results)
    
    # Make predictions
    predictions = model.predict(train_data, steps=len(test_data))
    print("Predictions shape:", predictions.shape)
    
    # Evaluate performance
    evaluation = model.evaluate(train_data, test_data)
    print("Evaluation metrics:", evaluation)