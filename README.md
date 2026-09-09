# TXGuard: AI Risk Manager and Fraud Detection for Financials

> **Note:** The original repo had to be deleted due to sloppy code. I have created a fresh repo with the same name to keep the commit history clean, modular, and reflective of the final production pipeline built during this hackathon.

TXGuard is a Python-based fraud detection system for highly imbalanced card-transaction classification, optimized for operational precision-recall tradeoffs and financial impact.

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- Raw dataset files (including `creditcard.csv`) are intentionally excluded via `.gitignore` following standard ML repository practices.
- Place `creditcard.csv` in the local `data/` directory before training.

## Setup

1. Clone the repository.
2. Place `creditcard.csv` in `data/`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   python train.py
   ```

## System Architecture

![TxGuard System Architecture](assets/architecture_diagram.png)

## 🎥 Project Demo & Walkthrough

To add the demo video:
1. Open the repository on GitHub.
2. Edit `README.md` in the web editor.
3. Drag and drop the `.mp4` demo file directly into the editor.
4. Commit changes to embed the uploaded video link in the README.

## Experimental Iteration Log

| Iteration | Configuration | PR-AUC | Precision | Recall |
|---|---|---:|---:|---:|
| Baseline | PyTorch MLP | 0.6210 | 12.4% | 81.2% |
| Experiment 1 | Focal Loss | 0.7079 | 18.2% | 78.4% |
| Experiment 2 | PR-AUC Early Stopping + log1p | 0.7101 | 24.5% | 75.1% |
| Experiment 3 | WeightedRandomSampler | 0.7820 | 41.0% | 72.4% |
| Experiment 4 | Residual MLP + Mish + AdamW | 0.8232 | 85.37% | 71.43% |
| Final | TxGuard + Feature SMOTE + Cost-Sensitive Focal Loss + F2 Threshold 0.7177 | 0.8589 | 84.85% | 85.71% |

![TxGuard Evaluation Dashboard](assets/evaluation_dashboard.png)

## Held-Out Test Evaluation (56,962 Transactions)

- Captured **97/98 fraud cases** (**98.98% Effective Recall via Two-Tier Cascading Engine**)
- **Hard False Positives:** 18 cases (**0.032% Hard FPR**)
- **Net Financial Impact:** **₹320,900 Net Savings**
  - Gross Fraud Saved: ₹336,000
  - FP Insult Cost: -₹15,000
  - Compute: -₹100

## Implementation Architecture (Module Map)

- `data_setup.py`: Abstract dataset contracts, SMOTE synthesis, and PyTorch dataloader factory.
- `model_builder.py`: Deep Tabular Residual MLP architecture with skip connections and Mish activations.
- `engine.py`: Custom CostSensitiveFocalLoss, online hard negative mining, PR-AUC early stopping, and training loop.
- `utils.py`: Defensive model artifact saving pipelines.
- `train.py`: CLI orchestration script with OneCycleLR scheduling.
- `app.py`: Integration endpoints for transaction scoring.

## Repository Structure

```text
txguard/
├─ app.py
├─ data_setup.py
├─ engine.py
├─ model_builder.py
├─ train.py
├─ utils.py
├─ data/
│  └─ creditcard.csv (local only; gitignored)
├─ assets/
│  ├─ architecture_diagram.png
│  └─ evaluation_dashboard.png
├─ requirements.txt
└─ README.md
```
