# src/evaluation/generate_thesis_results.py
"""
Thesis Phase 3: Robustness & Impact Analysis
Generates tables and figures for the Thesis Defense
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_analysis(results_path: str = 'experiments/results/results_multi_item.csv',
                     output_dir: str = 'figures/thesis'):
    """
    Generate all thesis analysis artifacts
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load Results
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        logger.error(f"Results file not found at {results_path}")
        return

    # Filter Success
    df = df[df['status'] == 'success'].copy()
    if df.empty:
        logger.error("No successful results to analyze")
        return
        
    df['mae'] = pd.to_numeric(df['mae'])
    df['rmse'] = pd.to_numeric(df['rmse'])
    
    logger.info(f"Loaded {len(df)} records for analysis")
    
    # 1. Comparative Performance Table (Table 4.1 in Thesis)
    summary = df.groupby('model')[['mae', 'rmse']].agg(['mean', 'std']).round(4)
    summary.to_csv(f'{output_dir}/model_summary_stats.csv')
    logger.info("\n📊 Model Performance Summary:")
    print(summary)
    
    # 2. Win/Loss Analysis (Robustness)
    # For each item and fold, find the best model
    best_models = df.loc[df.groupby(['item_id', 'fold'])['mae'].idxmin()]
    win_counts = best_models['model'].value_counts()
    win_rates = (win_counts / len(best_models) * 100).round(2)
    
    win_rates.to_csv(f'{output_dir}/win_rates.csv')
    logger.info("\n🏆 Win Rates (% of experiments where model was best):")
    print(win_rates)
    
    # Plot Win Rates
    plt.figure(figsize=(8, 5))
    sns.barplot(x=win_counts.index, y=win_counts.values, palette='viridis')
    plt.title('Winning Frequency by Model (Lower MAE is better)')
    plt.ylabel('Number of Wins')
    plt.savefig(f'{output_dir}/win_rates_plot.png')
    plt.close()
    
    # 3. Distribution Analysis (Boxplots) - Showing Stability
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model', y='mae', palette='Set3')
    plt.title('MAE Distribution across all Items and Folds')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/mae_distribution_boxplot.png')
    plt.close()
    
    # 4. Pairwise Comparison: Stacking vs. Best Single Model (e.g., LightGBM)
    # Pivot data to get models side-by-side
    pivot_df = df.pivot_table(index=['item_id', 'fold'], columns='model', values='mae')
    
    if 'StackingEnsemble' in pivot_df.columns and 'LightGBM' in pivot_df.columns:
        stacking_wins = np.sum(pivot_df['StackingEnsemble'] < pivot_df['LightGBM'])
        total_comparisons = len(pivot_df.dropna())
        
        logger.info(f"\n⚔️ Head-to-Head: Stacking vs LightGBM")
        logger.info(f"Stacking wins: {stacking_wins}/{total_comparisons} ({stacking_wins/total_comparisons*100:.1f}%)")
        
        # Improvement Percentage
        improvement = (pivot_df['LightGBM'] - pivot_df['StackingEnsemble']) / pivot_df['LightGBM'] * 100
        avg_improvement = improvement.mean()
        logger.info(f"Average Improvement over LightGBM: {avg_improvement:.2f}%")
        
    # 5. Business Impact Estimation
    # Assumptions: 
    # - Holding Cost = $1 per unit per day
    # - Stockout Cost = $5 per unit (lost margin)
    # - Error implies safety stock buffer needed. Higher MAE -> Higher Buffer -> Higher Cost.
    # Simple proxy: Cost ~ (MAE * 1.5) * Holding_Cost + (RMSE * 0.5) * Stockout_Cost ?? 
    # Let's use a simpler proxy: Total Cost = Sum(Abs(Error)) * Avg_Cost_Per_Error
    # Avg_Cost_Per_Error = 0.5 * Holding + 0.5 * Stockout = 3$
    
    cost_per_error_unit = 3.0 # Weighted average of over/under prediction costs
    
    df['estimated_cost'] = df['mae'] * cost_per_error_unit * 30 # For 30 days horizon
    
    cost_summary = df.groupby('model')['estimated_cost'].sum().sort_values()
    cost_summary.to_csv(f'{output_dir}/estimated_business_cost.csv')
    
    logger.info("\n💰 Estimated Monthly Inventory Cost (Proxy Calculation):")
    print(cost_summary)
    
    plt.figure(figsize=(8, 5))
    cost_summary.plot(kind='bar', color='salmon')
    plt.title('Total Estimated Inventory Cost (Lower is Better)')
    plt.ylabel('Cost ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/business_impact_plot.png')
    plt.close()
    
    logger.info(f"\n✅ Analysis Completed. Figures saved to {output_dir}/")

if __name__ == "__main__":
    generate_analysis()
