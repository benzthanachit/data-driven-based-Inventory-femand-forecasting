# generate_figures.py (Updated for Stacking Ensemble)
"""
สร้างรูปภาพสำหรับรายงานงานวิจัย Hybrid Ensemble Demand Forecasting (เวอร์ชันล่าสุด)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

# Ensure src path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils.data_utils import load_m5_data, prepare_m5_data_for_models
from models.base.sarimax_model import SARIMAXModel
from models.base.lstm_model import LSTMModel
from models.base.lightgbm_model import LightGBMModel
from models.ensemble.weighted_ensemble import WeightedAverageEnsemble
from sklearn.linear_model import Ridge # For loading Stacking meta-learner if needed

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figures directory
FIGURES_DIR = Path('figures_v2')
FIGURES_DIR.mkdir(exist_ok=True)

# --- Helper function to load JSON safely ---
def load_json_safely(filepath):
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warning: Could not find {filepath}. Some plots may be skipped.")
        return None

# --- FIGURE GENERATION FUNCTIONS (Updated and New) ---

# === COPY AND REPLACE THIS ENTIRE FUNCTION in generate_figures.py ===

# === COPY AND REPLACE THIS ENTIRE FUNCTION in generate_figures.py ===

def plot_methodology_flowchart_updated():
    """
    รูปที่ 7 (แก้ไขครั้งสุดท้าย): Methodology Flowchart ที่แสดงทั้ง 2 แนวทางของ Ensemble
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Layout that shows two parallel experimental paths
    boxes = [
        {'name': 'M5 Dataset\n(HOBBIES_1_001)', 'pos': (0.5, 0.95), 'color': 'lightblue'},
        {'name': 'Data Preprocessing\n& Feature Engineering', 'pos': (0.5, 0.85), 'color': 'lightgray'},
        {'name': 'Train/Val/Test Split', 'pos': (0.5, 0.75), 'color': 'lightgray'},
        
        # Base Models
        {'name': 'SARIMAX', 'pos': (0.2, 0.6), 'color': 'lightcoral'},
        {'name': 'LSTM', 'pos': (0.5, 0.6), 'color': 'lightcoral'},
        {'name': 'LightGBM', 'pos': (0.8, 0.6), 'color': 'lightcoral'},
        
        # Two Ensemble Paths
        {'name': 'Path A: Weighted Average Ensemble\n(Evaluated with 5-Fold CV for Stability)', 'pos': (0.25, 0.4), 'color': 'lightgreen'},
        {'name': 'Path B: Stacking Ensemble\n(Trained with 5-Fold CV for Accuracy)', 'pos': (0.75, 0.4), 'color': 'gold'},
        
        # Final Step
        {'name': 'Final Comparison\n& Conclusion', 'pos': (0.5, 0.2), 'color': 'khaki'}
    ]
    
    for box in boxes:
        rect = plt.Rectangle((box['pos'][0]-0.15, box['pos'][1]-0.04), 0.30, 0.08, 
                           facecolor=box['color'], edgecolor='black', alpha=0.8, zorder=5)
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['name'], ha='center', va='center', 
               fontsize=9, fontweight='bold', wrap=True, zorder=6)
    
    # Arrows reflecting the full story
    arrows = [
        # Top part
        ((0.5, 0.91), (0.5, 0.89)), # Dataset -> Preproc
        ((0.5, 0.81), (0.5, 0.79)), # Preproc -> Split
        ((0.5, 0.71), (0.2, 0.64)), # Split -> SARIMAX
        ((0.5, 0.71), (0.5, 0.64)), # Split -> LSTM
        ((0.5, 0.71), (0.8, 0.64)), # Split -> LGBM
        
        # Path A: Base models feeding into Weighted Average
        ((0.2, 0.56), (0.25, 0.44)), # SARIMAX -> Weighted
        ((0.5, 0.56), (0.25, 0.44)),  # LSTM -> Weighted
        ((0.8, 0.56), (0.25, 0.44)), # LightGBM -> Weighted

        # Path B: Base models feeding into Stacking
        ((0.2, 0.56), (0.75, 0.44)),  # SARIMAX -> Stacking
        ((0.5, 0.56), (0.75, 0.44)), # LSTM -> Stacking
        ((0.8, 0.56), (0.75, 0.44)),  # LightGBM -> Stacking
        
        # Both paths lead to the final comparison
        ((0.25, 0.36), (0.45, 0.24)), # Weighted -> Final
        ((0.75, 0.36), (0.55, 0.24))  # Stacking -> Final
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1)
    ax.axis('off')
    ax.set_title('Complete Research Methodology Flowchart', fontweight='bold', fontsize=16, pad=20)
    
    plt.savefig(FIGURES_DIR / '07_methodology_flowchart_complete.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 7 (Complete): Full Methodology Flowchart saved")

def plot_final_model_comparison():
    """
    [NEW] รูปที่ 3 (ปรับปรุง): Final Model Performance Comparison on Test Set
    """
    step1 = load_json_safely('experiments/results/step1_report.json')
    step2 = load_json_safely('experiments/results/step2_ensemble_report.json')
    step4 = load_json_safely('experiments/results/step4_stacking_report.json')

    if not all([step1, step2, step4]):
        print("❌ Skipping Figure 3 (Updated): Missing one or more report files.")
        return

    data = {
        'LightGBM': step1['performance_comparison']['lightgbm']['mae'],
        'Weighted Avg': step2['test_mae'],
        'Stacking': step4['test_mae']
    }
    
    models = list(data.keys())
    mae_scores = list(data.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(models, mae_scores, color=['lightgreen', 'lightcoral', 'skyblue'], alpha=0.9)
    ax.set_title('Final Model Performance on Test Set', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.grid(axis='y', alpha=0.4)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 3 (Updated): Final Model Comparison saved")

def plot_forecast_visualization_updated():
    """
    รูปที่ 6 (ปรับปรุง): Forecast Visualization with REAL Predictions
    """
    try:
        # --- This part regenerates predictions for plotting ---
        print("   - Generating predictions for final forecast plot...")
        train_df, val_df, test_df = prepare_m5_data_for_models("HOBBIES_1_001", "CA_1")
        y_test = test_df['demand'].values

        # Base model predictions
        sar = SARIMAXModel(); sar.load_model('models/saved/sarimax_model.pkl')
        lstm = LSTMModel(); lstm.load_model('models/saved/lstm_model')
        lgbm = LightGBMModel(); lgbm.load_model('models/saved/lightgbm_model')

        p_test_sar = sar.predict(steps=len(test_df))
        lstm_input = np.concatenate([train_df['demand'].values, val_df['demand'].values])
        p_test_lstm = lstm.predict(lstm_input, steps=len(test_df))
        
        full = pd.concat([train_df, val_df, test_df], ignore_index=True)
        feat = lgbm.create_features(full, 'demand')
        test_feat = feat.iloc[len(train_df)+len(val_df):].drop(columns=['date','demand'])
        p_test_lgbm = lgbm.predict(test_feat)

        # Weighted Average Ensemble prediction
        step2_report = load_json_safely('experiments/results/step2_ensemble_report.json')
        if step2_report:
            weights_avg = step2_report['weights']
            preds_test_avg = {'sarimax': p_test_sar, 'lstm': p_test_lstm, 'lightgbm': p_test_lgbm}
            ens_avg = WeightedAverageEnsemble()
            ens_avg.weights_ = np.array([weights_avg.get(name, 0) for name in ['sarimax', 'lstm', 'lightgbm']])
            ens_avg.model_names_ = ['sarimax', 'lstm', 'lightgbm']
            ens_avg.is_fitted = True
            p_test_avg = ens_avg.predict(preds_test_avg)
        else:
            p_test_avg = np.zeros_like(y_test) # Placeholder

        # Stacking Ensemble prediction
        step4_report = load_json_safely('experiments/results/step4_stacking_report.json')
        if step4_report:
            meta_coef = np.array(step4_report['meta_learner_coef'])
            X_test_meta = np.column_stack([p_test_sar, p_test_lstm, p_test_lgbm])
            # Simple prediction assuming Ridge has no intercept or it's negligible for the plot
            p_test_stack = X_test_meta.dot(meta_coef)
        else:
            p_test_stack = np.zeros_like(y_test) # Placeholder
            
        # --- Plotting ---
        plot_data = test_df.tail(60).copy()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        ax.plot(plot_data['date'], plot_data['demand'], 'ko-', lw=2, ms=4, label='Actual', alpha=0.8, zorder=5)
        ax.plot(plot_data['date'], p_test_lgbm[-60:], '--', lw=1.5, label='LightGBM', alpha=0.7)
        ax.plot(plot_data['date'], p_test_avg[-60:], ':', lw=2, label='Weighted Avg', alpha=0.8, color='purple')
        ax.plot(plot_data['date'], p_test_stack[-60:], '-', lw=2.5, label='Stacking Ensemble', color='red', alpha=0.9)
        
        ax.set_title('Final Forecast Comparison: Last 60 Days of Test Period', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Demand (Units)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / '06_forecast_visualization_updated.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Figure 6 (Updated): Forecast Visualization with real predictions saved")
    
    except Exception as e:
        print(f"❌ Failed to generate Figure 6 (Updated). Error: {e}. Check if models are trained and reports exist.")

# --- Main execution ---
def main():
    """Generate all figures for the research report"""
    print("🎨 Generating updated figures for the report...")
    print("="*60)
    
    # These figures are still relevant and do not need changes
    # plot_time_series_overview()
    # plot_data_split_diagram()
    # plot_cv_results_boxplot()
    # plot_ensemble_weights_analysis()
    # plot_wilcoxon_test_results()

    # --- Updated/New Figures ---
    plot_final_model_comparison()        # NEW: Compares final model performance
    plot_forecast_visualization_updated()# UPDATED: Uses real predictions
    plot_methodology_flowchart_updated() # UPDATED: Includes Stacking
    
    print("="*60)
    print("✅ All necessary figures generated successfully!")
    print(f"📁 Saved to: {FIGURES_DIR}/ directory")
    print("\n📋 Recommended Figure List for Updated Report:")
    print("1. Figure '01_time_series_overview.png' -> Data Description")
    print("2. Figure '07_methodology_flowchart_updated.png' -> Methodology Overview")
    print("3. Figure '03_final_model_comparison.png' -> Main Result Summary")
    print("4. Figure '06_forecast_visualization_updated.png' -> Forecast Quality")
    print("5. Figure '04_cv_results_boxplot.png' -> Stability Analysis (from CV)")
    print("6. Figure '09_wilcoxon_test_results.png' -> Statistical Test (for CV)")

if __name__ == "__main__":
    main()