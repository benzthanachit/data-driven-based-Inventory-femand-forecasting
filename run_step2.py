# run_step2.py - Fixed version for model loading issues

"""
Step 2: Weighted Average Ensemble Implementation
Building on successful Step 1 results to create optimized ensemble
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add src and experiments to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
experiments_path = current_dir / "experiments"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(experiments_path))

try:
    # Import models
    from models.base.sarimax_model import SARIMAXModel
    from models.base.lstm_model import LSTMModel
    from models.base.lightgbm_model import LightGBMModel
    from models.ensemble.weighted_ensemble import WeightedAverageEnsemble

    # Import utilities
    from utils.data_utils import load_data
    from experiments.configs.ensemble_configs import get_ensemble_config

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in place and Step 1 was completed successfully")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/step2_execution.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_model_files():
    """Check what model files actually exist"""
    print("🔍 Checking model files...")
    model_dir = "models/saved"
    if not os.path.exists(model_dir):
        print(f"❌ Directory {model_dir} does not exist!")
        return []
    files = os.listdir(model_dir)
    print(f"Files found in {model_dir}:")
    for f in files:
        print(f"  - {f}")
    return files

def step2_load_models_and_data():
    """Step 2A: Load trained models and data from Step 1 - FIXED VERSION"""
    print("\n" + "="*60)
    print("🔸 STEP 2A: LOADING MODELS AND DATA")
    print("="*60)

    # Load configuration
    config = get_ensemble_config('full_optimization')
    model_config = config['model_loading']

    # Check existing model files
    model_files = check_model_files()
    if not model_files:
        print("❌ No model files found. Please run Step 1 first.")
        return None, None, None, None, None

    try:
        # Load data
        print("📊 Loading processed data...")
        train_df = pd.read_csv(model_config['data_paths']['train'])
        val_df = pd.read_csv(model_config['data_paths']['val'])
        test_df = pd.read_csv(model_config['data_paths']['test'])
        print(f"✅ Data loaded: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

        # Load models
        print("🤖 Loading trained models...")
        models = {}

        # Load SARIMAX via pickle
        try:
            print("Loading SARIMAX model...")
            import pickle
            with open('models/saved/sarimax_model.pkl', 'rb') as f:
                sarimax_data = pickle.load(f)
            sarimax_model = SARIMAXModel()
            sarimax_model.fitted_model = sarimax_data['fitted_model']
            sarimax_model.training_data = sarimax_data['training_data']
            sarimax_model.exog_data = sarimax_data['exog_data']
            sarimax_model.is_trained = True
            params = sarimax_data.get('params', {})
            for attr in ['order','seasonal_order','trend','enforce_stationarity','enforce_invertibility']:
                if attr in params:
                    setattr(sarimax_model, attr, params[attr])
            models['sarimax'] = sarimax_model
            print("✅ SARIMAX model loaded")
        except Exception as e:
            print(f"⚠️  SARIMAX loading failed: {e}")
            print("Continuing without SARIMAX...")

        # Load LSTM
        try:
            print("Loading LSTM model...")
            lstm_model = LSTMModel()
            lstm_model.load_model('models/saved/lstm_model')
            models['lstm'] = lstm_model
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"⚠️  LSTM loading failed: {e}")
            print("Continuing without LSTM...")

        # Load LightGBM
        try:
            print("Loading LightGBM model...")
            lightgbm_model = LightGBMModel()
            lightgbm_model.load_model('models/saved/lightgbm_model')
            models['lightgbm'] = lightgbm_model
            print("✅ LightGBM model loaded")
        except Exception as e:
            print(f"⚠️  LightGBM loading failed: {e}")
            print("Continuing without LightGBM...")

        if not models:
            print("❌ No models could be loaded!")
            return None, None, None, None, None

        print(f"✅ Successfully loaded {len(models)} models: {list(models.keys())}")
        return models, train_df, val_df, test_df, config

    except Exception as e:
        print(f"❌ Failed to load models and data: {e}")
        return None, None, None, None, None

def step2_generate_predictions(models, train_df, val_df, test_df):
    """Step 2B: Generate predictions from all models"""
    print("\n" + "="*60)
    print("🔸 STEP 2B: GENERATING MODEL PREDICTIONS")
    print("="*60)
    try:
        predictions = {}
        y_true = test_df['demand'].values

        if 'sarimax' in models:
            print("🔮 Generating SARIMAX predictions...")
            pred = models['sarimax'].predict(steps=len(test_df))
            predictions['sarimax'] = pred
            print(f"✅ SARIMAX predictions: {len(pred)}")

        if 'lstm' in models:
            print("🔮 Generating LSTM predictions...")
            td = train_df['demand'].values
            pred = models['lstm'].predict(td, steps=len(test_df))
            predictions['lstm'] = pred
            print(f"✅ LSTM predictions: {len(pred)}")

        if 'lightgbm' in models:
            print("🔮 Generating LightGBM predictions...")
            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            feature_df = models['lightgbm'].create_features(full_df, 'demand')
            start = len(train_df)+len(val_df)
            tf = feature_df.iloc[start:]
            cols = [c for c in tf.columns if c not in ['date','demand']]
            X_test = tf[cols].dropna()
            pred = models['lightgbm'].predict(X_test)
            if len(pred)<len(test_df):
                pred = np.pad(pred,(0,len(test_df)-len(pred)),'edge')
            elif len(pred)>len(test_df):
                pred = pred[:len(test_df)]
            predictions['lightgbm'] = pred
            print(f"✅ LightGBM predictions: {len(pred)}")

        print("\n📊 Individual Performance:")
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        for name,p in predictions.items():
            mae = mean_absolute_error(y_true,p)
            rmse = np.sqrt(mean_squared_error(y_true,p))
            print(f"   {name.upper():>10}: MAE={mae:.4f}, RMSE={rmse:.4f}")

        return predictions, y_true
    except Exception as e:
        print(f"❌ Prediction generation failed: {e}")
        return None, None

def step2_optimize_ensemble(predictions,y_true,config):
    """Step 2C: Optimize ensemble weights"""
    print("\n" + "="*60)
    print("🔸 STEP 2C: OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*60)
    if len(predictions)<2:
        print("❌ Need at least 2 models for ensemble.")
        return None,None
    ec = config['weighted_ensemble']
    ensemble = WeightedAverageEnsemble(
        optimization_method=ec['optimization_method'],
        loss_function=ec['loss_function'],
        weights_bounds=ec['weights_bounds'],
        normalize_weights=ec['normalize_weights'],
        random_state=ec['random_state']
    )
    print("🎯 Optimizing weights...")
    res = ensemble.fit(predictions,y_true)
    print(f"✅ Best {ec['loss_function']}: {res['best_score']:.6f}")
    print("\nWeights:")
    for m,w in res['weights'].items():
        print(f"  {m}: {w:.3f}")
    return ensemble,res

def step2_evaluate_ensemble(ensemble,predictions,y_true):
    """Step 2D: Evaluate ensemble performance"""
    print("\n" + "="*60)
    print("🔸 STEP 2D: EVALUATING ENSEMBLE PERFORMANCE")
    print("="*60)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    pred = ensemble.predict(predictions)
    mae = mean_absolute_error(y_true,pred)
    rmse = np.sqrt(mean_squared_error(y_true,pred))
    print(f"📊 Ensemble MAE={mae:.4f}, RMSE={rmse:.4f}")
    return pred, {'ensemble':{'mae':mae,'rmse':rmse}}

def step2_save_results(ensemble,fit_results,perf,config):
    """Step 2E: Save ensemble model and report"""
    print("\n" + "="*60)
    print("🔸 STEP 2E: SAVING RESULTS")
    print("="*60)
    rc = config['results']
    Path(rc['save_directory']).mkdir(parents=True,exist_ok=True)
    Path('models/saved').mkdir(parents=True,exist_ok=True)
    ensemble.save_ensemble(rc['ensemble_model_path'])
    report = {
        'timestamp':datetime.now().isoformat(),
        'optimization':fit_results,
        'performance':perf,
        'weights':ensemble.get_weights()
    }
    with open(f"{rc['save_directory']}/{rc['report_filename']}",'w') as f:
        json.dump(report,f,indent=2,default=str)
    print(f"📄 Report saved: {rc['save_directory']}/{rc['report_filename']}")
    return report

def main():
    print("🚀 STEP 2: WEIGHTED AVERAGE ENSEMBLE IMPLEMENTATION")
    setup_logging()
    models,train_df,val_df,test_df,config = step2_load_models_and_data()
    if models is None: sys.exit(1)
    preds,y_true = step2_generate_predictions(models,train_df,val_df,test_df)
    if preds is None: sys.exit(1)
    ensemble,fit_res = step2_optimize_ensemble(preds,y_true,config)
    if ensemble is None: sys.exit(1)
    _,perf = step2_evaluate_ensemble(ensemble,preds,y_true)
    report = step2_save_results(ensemble,fit_res,perf,config)
    print("\n✅ STEP 2 COMPLETED SUCCESSFULLY!")
    sys.exit(0 if report else 1)

if __name__ == "__main__":
    main()