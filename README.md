# TXGuard: AI Risk Manager and Fraud Detection for Financials

> **Note:** The original repo had to be deleted due to sloppy code. I have created a fresh repo with the same name to show the actual production pipeline built for the Razorpay buildathon.

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

## Project Demo & Walkthrough

To add the demo video:
1. Open the repository on GitHub.
2. Edit `README.md` in the web editor.
3. Drag and drop the `.mp4` demo file directly into the editor.
4. Commit changes to embed the uploaded video link in the README.

## 🎢 The Real Journey (Behind the Scenes)

If you look at typical ML repos, everything looks like a smooth upward curve. Ours wasn't. It was a complete roller coaster.

We started with a basic 3-layer MLP and default BCE loss—precision dropped to 12% instantly because the model just guessed non-fraud to keep the loss low. So we slapped on Focal Loss with alpha=0.25 and gamma=2.0. Validation loss started looking great, but when we checked PR-AUC, performance had actually fallen off a cliff. The focal loss was dampening easy sample loss so hard that early stopping was triggering on misleading numbers.

We ditched validation loss early stopping and switched strictly to tracking Validation PR-AUC. Then we tried making the model deeper. Bad idea—standard ReLUs killed off gradients, neurons died, and recall dropped to 71%. We had to rip that out, swap in Mish activations, and add skip connections (Residual blocks) just to get feature signal flowing again.

Then came the threshold nightmare: lowering the cutoff to catch missing fraudsters spiked false positives through the roof. Bumping gamma to 4.0 and adding an 8x False Negative penalty in a custom CostSensitiveFocalLoss finally stabilized things, but a single cutoff was still an impossible compromise. That’s why we ended up with the Two-Tier Cascading Engine (Auto-Approve, 2FA Step-Up, Auto-Block)—it was the only way to catch borderline fraud without slamming the door on real customers.

![TxGuard Evaluation Dashboard](assets/evaluation_dashboard.png)

### Real-World Business Impact (Razorpay Buildathon Benchmark)

Instead of evaluating on vanity accuracy metrics, we tested the engine against a simulated slice of real-world processing volume (~57,000 transactions) under strict BFSI rules (keeping customer false blocks well under 1%).

- **98.98% Total Fraud Intercepted:** The combined Auto-Block + 2FA Step-Up engine caught 97 out of 98 fraud attempts. The 2FA layer seamlessly caught borderline attacks without needing hard declines.
- **Near-Zero Customer Insults (0.032% FPR):** Out of tens of thousands of legitimate transactions, only 18 real users faced a hard block.
- **Bottom-Line Financial Ledger:**
  * **Gross Fraud Stopped:** ~₹3,36,000 in saved chargebacks and stolen funds.
  * **Customer Friction Cost:** ~₹15,000 (estimated cost impact of false alarms/support tickets).
  * **Net Value Delivered:** **₹320,900 net savings** per test batch after accounting for compute overhead.

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
