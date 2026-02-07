# Research Proposal: Hybrid Ensemble for Inventory Forecasting
> **Target**: Master's Thesis Defense & International Publication (Scopus Indexed)
> **Topic**: Data-Driven Inventory Demand Forecasting using Hybrid Ensemble Learning
> **Current Status**: Final Validation Phase (Multi-Item Scope)

## Goal Description
To conduct rigorous research and develop a robust forecasting system that outperforms traditional models (SARIMA) and modern ML models (LSTM, LightGBM). The core contribution is a **Stacking/Blending Ensemble** strategy validated on a diverse set of items from the **M5 Forecasting Dataset** to prove generalizability.

## User Review Required
> [!IMPORTANT]
> **Data Access**: Ensure M5 competition data (`sales_train_validation.csv`) is available.
> **Scope Update**: Moving from Single-Item POC to **Multi-Item Validation (20-50 SKUs)**.

## Proposed Research Methodology

### 1. Data Side (Data Preparation & Engineering)
- **Dataset**: M5 Forecasting - Accuracy (Walmart).
- **Scope Expansion**: Instead of a single time series, we select a representative subset of **20-50 SKUs** from the `HOBBIES` category, covering:
    -   High Volume Items
    -   Low Volume / Intermittent Items (Hard to forecast)
- **Preprocessing**:
    -   **Dynamic Loading**: Pipeline must handle `item_id` selection dynamically.
    -   **Feature Engineering**: Lag features (7, 14, 28 days), Rolling Stats, and Exogenous variables (SNAP, Price) applied per item.

### 2. Modeling Side (The Hybrid Approach)
We propose a multi-stage stacking regressor.
- **Base Learners**:
    -   **SARIMAX**: Captures linear seasonality.
    -   **LSTM**: Captures non-linear temporal dependencies.
    -   **LightGBM**: Captures complex feature interactions.
- **Ensemble Layer (Meta-Learner)**:
    -   **Stacking**: Use Ridge Regression (or similar) to learn optimal weights based on Out-of-Fold (OOF) predictions.
    -   **Hypothesis**: The ensemble should demonstrate higher stability (lower Std Dev of error) across the diverse SKU set compared to single models.

### 3. Verification & Evaluation (For Defense & Publication)
To ensure the thesis is defensible and publishable:
-   **Validation Strategy**:
    -   **5-Fold Rolling Origin Cross-Validation (ROCV)**: Applied to *each* of the selected SKUs.
-   **Key Metrics**:
    -   **Accuracy**: MAE, RMSE.
    -   **Robustness**: Win/Loss Ratio (Ensemble vs. Single Best) and MAE Distribution (Boxplots).
    -   **Business Impact**: Proxy calculation for Inventory Costs (Holding vs. Stockout) to demonstrate practical value.
-   **Statistical Tests**:
    -   **Wilcoxon Signed-Rank Test**: To test significance across the multiple items.

## Verification Plan

### Automated Tests (Batch Pipeline)
- Run `src/utils/m5_data_processor.py` to verify multi-item loading.
- Implement `run_batch_experiment.py` to loop through SKUs and save logs.
- Generate `results_multi_item.csv` for analysis.

### Manual Verification
1.  **Distribution Check**: Ensure MAE variance across items is analyzed.
2.  **Fail Analysis**: Investigate items where Ensemble performs worse than baseline.