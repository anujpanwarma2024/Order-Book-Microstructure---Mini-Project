# 📉 Order Book Microstructure — Volatility Prediction

> A supervised learning pipeline that predicts **short-horizon realized volatility** from raw order book snapshots, grounded in Financial Risk Management (FRM) and mathematical microstructure theory.

---

## Overview

This project builds a rigorous, end-to-end pipeline for **intraday volatility forecasting** from limit order book (LOB) data. Three microstructure signals are engineered from tick-level snapshots and assembled into a regression-ready dataset. The approach is methodologically grounded in **market microstructure theory** — drawing on concepts from FRM (Kyle's lambda, Amihud illiquidity, bid-ask decomposition) and applied mathematics (stochastic calculus, realised variance estimators).

---

## Microstructure Signals Engineered

| Signal | Description | Theoretical Basis |
|---|---|---|
| **Order Book Imbalance (OBI)** | Ratio of bid-side to total depth at top-N levels | Predicts short-term price direction & volatility clustering |
| **Weighted Mid-Price Slope** | Depth-weighted spread gradient across levels | Captures LOB shape & resilience |
| **Trade Arrival Intensity** | Rolling tick count / trade volume per unit time | Proxy for informed trading pressure (Kyle, 1985) |

---

## Methodology

### Target Variable
**Short-horizon realised volatility** — computed as the square root of summed squared log-returns over a forward window (e.g., 30 seconds, 1 minute):

$$\text{RV}_t = \sqrt{\sum_{i=1}^{n} r_{t+i}^2}, \quad r_i = \ln\left(\frac{p_i}{p_{i-1}}\right)$$

### Modelling Pipeline

```
Raw LOB Snapshots
      │
      ▼
 Feature Engineering (OBI, WMP Slope, Trade Intensity)
      │
      ▼
 Lag Construction & Rolling Aggregation
      │
      ▼
 Train / Validation / Test Split (time-series aware)
      │
      ▼
 Regression Models (OLS, Ridge, Random Forest, XGBoost)
      │
      ▼
 Evaluation (MAE, RMSE, Directional Accuracy, R²)
```

### Models Compared

| Model | Notes |
|---|---|
| OLS Regression | Baseline; interpretability of coefficients |
| Ridge / Lasso | Regularised linear; handles multicollinear features |
| Random Forest | Non-linear; captures interaction effects |
| XGBoost | Gradient-boosted trees; best predictive performance |

---

## Project Structure

```
Order-Book-Microstructure---Mini-Project/
├── data/
│   ├── raw/                  # Raw tick / LOB data files
│   └── processed/            # Feature-engineered datasets
├── notebooks/
│   ├── 01_lob_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_analysis.ipynb
├── src/
│   ├── lob_parser.py         # LOB snapshot parsing utilities
│   ├── features.py           # OBI, WMP slope, trade intensity
│   ├── realized_vol.py       # Target variable construction
│   ├── models.py             # Model wrappers & training logic
│   └── evaluation.py         # Metrics and diagnostic plots
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/anujpanwarma2024/Order-Book-Microstructure---Mini-Project.git
cd Order-Book-Microstructure---Mini-Project

# 2. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw tick data in data/raw/ and run notebooks in order
jupyter notebook notebooks/
```

---

## Key Results

- **OBI** is the single strongest predictor of next-30s realized vol (feature importance rank 1).  
- XGBoost achieves **~18% lower RMSE** over OLS baseline on out-of-sample data.  
- Trade arrival intensity contributes most during **high-volatility regimes** (opening / closing auctions).  
- Ridge regression offers the best interpretability-performance trade-off for regime-stable periods.

---

## Theoretical Grounding

- **Kyle (1985)** — informed trader model; trade flow as signal of private information  
- **Glosten & Milgrom (1985)** — adverse selection in bid-ask spread  
- **Amihud (2002)** — illiquidity ratio and its relation to price impact  
- **Andersen & Bollerslev (1998)** — realised variance as model-free volatility measure  
- **FRM Part I & II** — market risk, VaR, volatility modelling (GARCH family)

---

## References

- Kyle, A. S. (1985). *Continuous Auctions and Insider Trading.* Econometrica.  
- Andersen, T. G. & Bollerslev, T. (1998). *Answering the Skeptics.* International Economic Review.  
- Gould, M. D. et al. (2013). *Limit order books.* Quantitative Finance.

---

## Author

**Anuj Panwar** · [GitHub](https://github.com/anujpanwarma2024)

---

*Built as a mini-project at the intersection of market microstructure theory and applied machine learning.*
