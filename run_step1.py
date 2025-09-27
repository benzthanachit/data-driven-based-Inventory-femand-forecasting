# Main runner script for Step 1: Base Models Implementation
"""
run_step1.py - Complete script to run Step 1 of the project
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from src.models.base.sarimax_model import SARIMAXModel
from src.models.base.lstm_model import LSTMModel
from src.models.base.lightgbm_model import LightGBMModel
from src.utils.data_utils import create_sample_data, split_time_series_data, check_data_quality
from experiments.configs.model_configs import get_config


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/step1_execution.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw', 'data/processed', 'data/synthetic',
        'models/saved', 'experiments/results', 
        'figures', 'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created necessary directories")


def step1_prepare_data():
    """Step 1A: Prepare synthetic data for testing"""
    print("\n" + "="*60)
    print("🔸 STEP 1A: DATA PREPARATION")
    print("="*60)
    
    # Create synthetic data
    print("📊 Creating synthetic data...")
    df = create_sample_data(
        n_samples=730,  # 2 years of data
        seasonal=True,
        random_seed=42
    )
    
    # Save synthetic data
    df.to_csv('data/synthetic/demand_data.csv', index=False)
    print(f"✅ Created and saved synthetic data: {df.shape[0]} samples")
    
    # Check data quality
    quality_report = check_data_quality(df, 'demand')
    print(f"📋 Data quality check completed")
    print(f"   - Date range: {quality_report['date_range'][0]} to {quality_report['date_range'][1]}")
    print(f"   - Demand range: {quality_report['target_stats']['min']:.2f} to {quality_report['target_stats']['max']:.2f}")
    print(f"   - Mean demand: {quality_report['target_stats']['mean']:.2f}")
    
    # Split data
    train_df, val_df, test_df = split_time_series_data(df, 'demand', 
                                                      train_ratio=0.7, 
                                                      val_ratio=0.15, 
                                                      test_ratio=0.15)
    
    # Save split data
    train_df.to_csv('data/processed/train_data.csv', index=False)
    val_df.to_csv('data/processed/val_data.csv', index=False)
    test_df.to_csv('data/processed/test_data.csv', index=False)
    
    print(f"✅ Data split completed: Train({len(train_df)}) | Val({len(val_df)}) | Test({len(test_df)})")
    
    return train_df, val_df, test_df


def step1_test_sarimax(train_df, val_df, test_df):
    """Step 1B: Test SARIMAX model"""
    print("\n" + "="*60)
    print("🔸 STEP 1B: SARIMAX MODEL TESTING")
    print("="*60)
    
    try:
        # Initialize SARIMAX model
        config = get_config('development')
        sarimax_config = config.SARIMAX_CONFIG
        
        print("🔧 Initializing SARIMAX model...")
        model = SARIMAXModel(
            order=sarimax_config['order'],
            seasonal_order=sarimax_config['seasonal_order']
        )
        
        # Train model
        print("🎯 Training SARIMAX model...")
        train_results = model.train(train_df['demand'])
        print(f"✅ SARIMAX training completed")
        print(f"   - AIC: {train_results['aic']:.4f}")
        print(f"   - BIC: {train_results['bic']:.4f}")
        
        # Test predictions
        print("🔮 Generating predictions...")
        test_predictions = model.predict(steps=len(test_df))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_df['demand'], test_predictions)
        rmse = np.sqrt(mean_squared_error(test_df['demand'], test_predictions))
        
        print(f"📊 SARIMAX Performance:")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        
        # Save model
        model.save_model('models/saved/sarimax_model.pkl')
        print("💾 SARIMAX model saved")
        
        return model, {'mae': mae, 'rmse': rmse}
        
    except Exception as e:
        print(f"❌ SARIMAX model failed: {str(e)}")
        return None, None


def step1_test_lstm(train_df, val_df, test_df):
    """Step 1C: Test LSTM model"""
    print("\n" + "="*60)
    print("🔸 STEP 1C: LSTM MODEL TESTING")
    print("="*60)
    
    try:
        # Initialize LSTM model
        config = get_config('development')
        lstm_config = config.LSTM_CONFIG
        
        print("🔧 Initializing LSTM model...")
        model = LSTMModel(
            sequence_length=lstm_config['sequence_length'],
            lstm_units=lstm_config['lstm_units'],
            dropout_rate=lstm_config['dropout_rate'],
            learning_rate=lstm_config['learning_rate']
        )
        
        # Train model
        print("🎯 Training LSTM model...")
        train_data = train_df['demand'].values
        train_results = model.train(
            train_data,
            epochs=lstm_config['epochs'],
            batch_size=lstm_config['batch_size'],
            verbose=1
        )
        print(f"✅ LSTM training completed")
        print(f"   - Final loss: {train_results['final_train_loss']:.6f}")
        print(f"   - Epochs trained: {train_results['epochs_trained']}")
        
        # Test predictions
        print("🔮 Generating predictions...")
        test_predictions = model.predict(train_data, steps=len(test_df))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test_df['demand'], test_predictions)
        rmse = np.sqrt(mean_squared_error(test_df['demand'], test_predictions))
        
        print(f"📊 LSTM Performance:")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        
        # Save model
        model.save_model('models/saved/lstm_model')
        print("💾 LSTM model saved")
        
        return model, {'mae': mae, 'rmse': rmse}
        
    except Exception as e:
        print(f"❌ LSTM model failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def step1_test_lightgbm(train_df, val_df, test_df):
    """Step 1D: Test LightGBM model"""
    print("\n" + "="*60)
    print("🔸 STEP 1D: LIGHTGBM MODEL TESTING")
    print("="*60)
    
    try:
        # Initialize LightGBM model
        config = get_config('development')
        lgbm_config = config.LIGHTGBM_CONFIG
        
        print("🔧 Initializing LightGBM model...")
        model = LightGBMModel(
            params=lgbm_config['params'],
            feature_engineering=lgbm_config['feature_engineering']
        )
        
        # Create features
        print("⚙️ Creating features...")
        train_features_df = model.create_features(train_df, 'demand')
        val_features_df = model.create_features(val_df, 'demand')
        test_features_df = model.create_features(test_df, 'demand')
        
        # Prepare training data
        feature_cols = [col for col in train_features_df.columns 
                       if col not in ['date', 'demand']]
        
        X_train = train_features_df[feature_cols].dropna()
        y_train = train_features_df.loc[X_train.index, 'demand']
        
        X_val = val_features_df[feature_cols].dropna()
        y_val = val_features_df.loc[X_val.index, 'demand']
        
        print(f"📊 Training data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Train model
        print("🎯 Training LightGBM model...")
        train_results = model.train(
            X_train, y_train,
            X_val, y_val,
            num_boost_round=lgbm_config['num_boost_round'],
            early_stopping_rounds=lgbm_config['early_stopping_rounds'],
            verbose_eval=50
        )
        print(f"✅ LightGBM training completed")
        print(f"   - Best iteration: {train_results['best_iteration']}")
        print(f"   - Number of features: {train_results['n_features']}")
        
        # Test predictions
        print("🔮 Generating predictions...")
        X_test = test_features_df[feature_cols].dropna()
        test_predictions = model.predict(X_test)
        
        # Align test data (in case of dropped NaN)
        y_test = test_features_df.loc[X_test.index, 'demand']
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        print(f"📊 LightGBM Performance:")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        
        # Show feature importance
        importance = model.get_feature_importance().head(5)
        print("🔝 Top 5 features:")
        for _, row in importance.iterrows():
            print(f"   - {row['feature']}: {row['importance']:.2f}")
        
        # Save model
        model.save_model('models/saved/lightgbm_model')
        print("💾 LightGBM model saved")
        
        return model, {'mae': mae, 'rmse': rmse}
        
    except Exception as e:
        print(f"❌ LightGBM model failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def step1_generate_report(results):
    """Step 1E: Generate final report"""
    print("\n" + "="*60)
    print("🔸 STEP 1E: GENERATING FINAL REPORT")
    print("="*60)
    
    # Create results summary
    report = {
        'timestamp': datetime.now().isoformat(),
        'models_tested': [],
        'performance_comparison': {},
        'best_model': None,
        'status': 'completed'
    }
    
    # Compile results
    for model_name, performance in results.items():
        if performance is not None:
            report['models_tested'].append(model_name)
            report['performance_comparison'][model_name] = performance
    
    # Find best model (based on MAE)
    if report['performance_comparison']:
        best_model = min(report['performance_comparison'].items(), 
                        key=lambda x: x[1]['mae'])
        report['best_model'] = {
            'name': best_model[0],
            'mae': best_model[1]['mae'],
            'rmse': best_model[1]['rmse']
        }
    
    # Save report
    import json
    with open('experiments/results/step1_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("📋 STEP 1 EXECUTION SUMMARY")
    print("-" * 30)
    print(f"Models successfully tested: {len(report['models_tested'])}")
    
    if report['performance_comparison']:
        print("\n📊 Performance Comparison:")
        for model_name, metrics in report['performance_comparison'].items():
            print(f"   {model_name.upper():>10}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        if report['best_model']:
            print(f"\n🏆 Best Model: {report['best_model']['name'].upper()}")
            print(f"    MAE: {report['best_model']['mae']:.4f}")
            print(f"    RMSE: {report['best_model']['rmse']:.4f}")
    
    print(f"\n💾 Report saved: experiments/results/step1_report.json")
    
    return report


def main():
    """Main execution function for Step 1"""
    print("🚀 STEP 1: BASE MODELS IMPLEMENTATION")
    print("=" * 80)
    print("Goal: จัดระเบียบและปรับปรุง SARIMAX, LSTM, และ LightGBM ให้พร้อมใช้งานในระบบ ensemble")
    print("=" * 80)
    
    # Setup
    logger = setup_logging()
    create_directories()
    
    # Step 1A: Prepare data
    train_df, val_df, test_df = step1_prepare_data()
    
    # Initialize results storage
    results = {}
    
    # Step 1B: Test SARIMAX
    sarimax_model, sarimax_results = step1_test_sarimax(train_df, val_df, test_df)
    results['sarimax'] = sarimax_results
    
    # Step 1C: Test LSTM
    lstm_model, lstm_results = step1_test_lstm(train_df, val_df, test_df)
    results['lstm'] = lstm_results
    
    # Step 1D: Test LightGBM
    lightgbm_model, lightgbm_results = step1_test_lightgbm(train_df, val_df, test_df)
    results['lightgbm'] = lightgbm_results
    
    # Step 1E: Generate report
    final_report = step1_generate_report(results)
    
    # Final status
    print("\n" + "="*80)
    print("✅ STEP 1 COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    successful_models = len([r for r in results.values() if r is not None])
    print(f"📊 Successfully implemented and tested {successful_models}/3 models")
    
    if successful_models >= 2:
        print("🎯 Ready for Step 2: Ensemble Implementation")
        print("\nNext steps:")
        print("1. Review model performance in experiments/results/step1_report.json")
        print("2. Check saved models in models/saved/")
        print("3. Proceed to implement Weighted Average Ensemble")
    else:
        print("⚠️  Some models failed. Please check logs and fix issues before proceeding.")
    
    print(f"\n📁 All files saved to:")
    print(f"   - Data: data/synthetic/, data/processed/")
    print(f"   - Models: models/saved/")
    print(f"   - Results: experiments/results/")
    print(f"   - Logs: logs/")
    
    return final_report


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    try:
        report = main()
        exit_code = 0 if len([r for r in report['performance_comparison'].values() if r is not None]) >= 2 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Step 1 execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)