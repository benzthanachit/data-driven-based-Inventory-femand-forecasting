# Hybrid Inventory Forecasting with Ensemble Learning

A comprehensive machine learning approach for inventory demand forecasting using ensemble methods combining SARIMAX, LSTM, and LightGBM models.

## 🎯 Project Overview

This project implements a hybrid ensemble learning framework for retail inventory demand forecasting. The system combines the strengths of statistical models (SARIMAX), deep learning (LSTM), and gradient boosting (LightGBM) to create more accurate and robust predictions.

## 🌟 Key Features

- **Multi-Model Ensemble**: Combines SARIMAX, LSTM, and LightGBM models
- **Advanced Preprocessing**: Automated feature engineering and data preparation
- **Flexible Architecture**: Easy to extend with new models and ensemble methods
- **Comprehensive Evaluation**: Multiple metrics and statistical significance testing
- **Production Ready**: Includes model persistence, logging, and configuration management
- **Research Oriented**: Designed for academic publication and reproducible research


## 📊 Model Architecture

### Base Models

1. **SARIMAX**: Seasonal AutoRegressive Integrated Moving Average with eXogenous variables
2. **LSTM**: Long Short-Term Memory neural network for sequence modeling
3. **LightGBM**: Gradient boosting framework optimized for efficiency

### Ensemble Methods

- Weighted Average Ensemble with optimization-based weight selection
- Cross-validation framework for robust model evaluation
- Support for multiple ensemble strategies (Stacking, Blending – coming soon)


## 🚀 Quick Start

### Installation

1. Clone the repository:

```
git clone https://github.com/benzthanachit/data-driven-based-Inventory-femand-forecasting.git
cd hybrid-inventory-forecasting
```

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Install the package in development mode:

```
pip install -e .
```


### Basic Usage

```python
from src.models.base.lstm_model import LSTMModel
from src.utils.data_utils import create_sample_data
from experiments.configs.model_configs import get_config

# Generate synthetic data
df = create_sample_data(n_samples=730, seasonal=True)

# Initialize model
config = get_config('development')
model = LSTMModel(sequence_length=config.LSTM_CONFIG['sequence_length'])

# Train model
train_data = df['demand'].values
model.train(train_data, epochs=config.LSTM_CONFIG['epochs'])

# Predict
predictions = model.predict(train_data, steps=30)
print(predictions)
```


## 📁 Project Structure

```
hybrid-inventory-forecasting/
├── src/
│   ├── models/
│   │   ├── base/            # SARIMAX, LSTM, LightGBM
│   │   └── ensemble/        # WeightedAverageEnsemble, Stacking
│   ├── utils/               # data_utils.py
│   ├── preprocessing/       # preprocessor.py
│   └── evaluation/          # evaluator.py
├── experiments/
│   ├── configs/             # model_configs.py
│   └── results/             # step1_report.json
├── data/
│   ├── raw/                 # raw data
│   ├── processed/           # train/val/test splits
│   └── synthetic/           # synthetic data
├── models/saved/            # saved models
├── figures/                 # plots
├── logs/                    # execution logs
├── tests/
│   ├── unit/
│   └── integration/         # test_pipeline.py
├── requirements.txt
├── run_step1.py             # Step 1 runner
└── README.md                # This file
```


## 📈 Model Performance (Example)

| Model | MAE | RMSE | MAPE |
| :-- | :-- | :-- | :-- |
| SARIMAX | 11.40 | 18.07 | 24.39% |
| LSTM | 8.51 | 14.05 | 18.32% |
| LightGBM | 5.02 | 6.84 | 12.03% |
| **Ensemble** | **4.85** | **6.45** | **11.20%** |

*Performance on synthetic data; results vary with real data.*

## 🔧 Configuration

Modify `experiments/configs/model_configs.py` to adjust hyperparameters, data splits, and ensemble settings.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{hybrid_inventory_forecasting_2025,
  title={Hybrid Ensemble Learning for Retail Inventory Demand Forecasting},
  author={Research Team},
  journal={International Journal of Production Economics},
  year={2025},
  note={Under Review}
}
```


## 🤝 Contributing

Contributions welcome! Please see `CONTRIBUTING.md` for guidelines.

## 📞 Contact

Email: research@example.com
GitHub: https://github.com/benzthanachit/data-driven-based-Inventory-femand-forecasting/issues

