# Research Task List: Hybrid Inventory Forecasting (Final Sprint for Defense)

This task list tracks the progress of your Master's research project, focusing on the "Data & Research" aspects for the final defense within 2 months.

## 📅 Phase 1: Multi-Item Pipeline Setup (Weeks 1-2)
*Focus: Refactor from Single-Item POC to Batch Processing*

- [x] **Acquire & Select Data**: Ensure `sales_train_validation.csv` is ready. Select **20-50 Target SKUs** from 'HOBBIES' category (Mix of High/Low/Intermittent).
- [x] **Refactor Data Processor**: Update `src/utils/m5_data_processor.py` to accept `item_id` as an argument for dynamic loading.
- [x] **Feature Engineering Validation**: Verify that lag features and rolling means are calculated correctly when switching between different items.
- [x] **Exploratory Data Analysis (EDA)**: Create a summary plot comparing demand patterns of the selected SKUs (to be used in Thesis Chapter 3).

## 📅 Phase 2: Modular Modeling & Batch Experimentation (Weeks 1-2)
*Focus: Standardization and Execution*

- [x] **Modularize Base Models**: Refactor `SARIMAX`, `LSTM`, and `LightGBM` code into callable functions (e.g., `train_lightgbm(data, config)`).
- [x] **Implement Stacking Class**: Create `src/models/ensemble/stacking_model.py` that implements the Ridge Regression Meta-learner.
- [x] **Develop Evaluation Loop**: Write the main loop to iterate through all selected SKUs using **5-Fold Rolling Origin Cross-Validation**.
- [x] **Error Handling**: Add `try-except` blocks to ensure the loop continues even if one SKU fails to converge.
- [x] **Run Batch Processing**: Execute the pipeline and save results to `results_multi_item.csv` (Columns: `item_id`, `model_type`, `fold`, `MAE`, `RMSE`).

## 📅 Phase 3: Robustness & Impact Analysis (Week 3)
*Focus: Proving the "Better" & "Stable" hypothesis*

- [x] **Win/Loss Analysis**: Calculate the percentage of items where Stacking outperforms LightGBM/SARIMAX.
- [x] **Distribution Analysis**: Generate Boxplots of MAE/RMSE across all items to demonstrate stability (Standard Deviation reduction).
- [x] **Business Impact Calculation**: Estimate potential savings (Holding Cost vs. Stockout Cost) comparing Stacking vs. Baseline.
- [x] **Fail Case Study**: Analyze items where Stacking failed to prepare answers for the defense Q&A.
- [x] **Improvement Action**: Investigate why Stacking (1.94) is losing to LightGBM (1.80) in multi-item (Likely LSTM noise).
    - *Result*: Tuned LSTM. Stacking Win Rate improved (e.g., wins 2/5 folds in Item 1), but aggregate MAE is still higher due to outlier failures in difficult items.
    - *Decision*: Proceed with "Honesty" approach in Thesis (Option A). Match results to specific business cases (e.g. Stacking is better for *stable* items).

## ⚡ Phase 3.5: Performance Optimization
- [x] **Enable GPU Support**: Configure LightGBM and TensorFlow to utilize the GPU for faster training.
    - *LightGBM*: Enabled `device_type='gpu'`. Test Passed.
    - *TensorFlow (LSTM)*: Current version (2.20) does not support Windows GPU. Created `GPU_SETUP_GUIDE.md` for manual downgrade instructions.
- [x] **Verify Acceleration**: Run a quick test to confirm GPU usage.
- [x] **Verify Acceleration**: Run a quick test to confirm GPU usage.

## 📅 Phase 4: Thesis Writing & Defense Prep (Weeks 4-8)
*Focus: Documentation and Presentation*

- [ ] **Write Methodology (Chapter 3)**: Document the Hybrid architecture and the Multi-item validation strategy.
- [ ] **Write Results (Chapter 4)**: Create comparative tables and "Improvement %" charts based on the multi-item run.
- [ ] **Update Discussion (Chapter 5)**: Discuss Generalizability (Robustness) and remove "Single item limitation".
- [ ] **Slide Deck Revision**: Update slides to show aggregate results (Boxplots) instead of single item plots.
- [ ] **Defense Rehearsal**: Practice explaining the trade-off between Computational Cost vs. Accuracy.

## 📝 Publication (Post-Defense)
- [ ] **Draft Scopus Paper**: Format results for target journal (e.g., Expert Systems with App), focusing on the "Generalizability" aspect proven in Phase 3.