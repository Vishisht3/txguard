import argparse
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from txguard.config import DROPOUT_RATE, HIDDEN_UNITS, INPUT_FEATURES, MODEL_PATH
from txguard.ml.engine import CostSensitiveFocalLoss, create_dataloaders, train
from txguard.ml.model import RiskScorerModel


def save_model(model: torch.nn.Module, target_dir: Union[str, Path], model_name: str) -> Path:
    """Saves PyTorch state_dict to a target directory."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name must end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    return model_save_path


def bootstrap_calibrated_model(
    model_path: Union[str, Path] = MODEL_PATH,
    input_features: int = INPUT_FEATURES,
    hidden_units: int = HIDDEN_UNITS,
) -> RiskScorerModel:
    """
    Loads the trained TxGuard RiskScorerModel from disk, or deterministically trains and saves
    a calibrated model on synthetic PayPal/card transaction distributions if no checkpoint exists.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_obj = Path(model_path)

    model = RiskScorerModel(
        input_features=input_features,
        hidden_units=hidden_units,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    if path_obj.exists():
        state_dict = torch.load(path_obj, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    torch.manual_seed(42)
    np.random.seed(42)

    n_legit = 700
    n_fraud = 300

    x_legit = np.random.normal(loc=0.0, scale=0.65, size=(n_legit, input_features)).astype(np.float32)
    x_legit[:, 0] = np.random.normal(loc=0.6, scale=0.4, size=n_legit)
    x_legit[:, 1] = np.random.normal(loc=-0.3, scale=0.4, size=n_legit)
    x_legit[:, 3] = np.random.normal(loc=-0.5, scale=0.4, size=n_legit)
    x_legit[:, 11] = np.random.normal(loc=0.5, scale=0.4, size=n_legit)
    x_legit[:, 13] = np.random.normal(loc=0.5, scale=0.4, size=n_legit)
    x_legit[:, 28] = np.random.normal(loc=-0.2, scale=0.5, size=n_legit)

    x_fraud = np.random.normal(loc=0.0, scale=0.9, size=(n_fraud, input_features)).astype(np.float32)
    x_fraud[:, 0] = np.random.normal(loc=-2.8, scale=0.7, size=n_fraud)
    x_fraud[:, 1] = np.random.normal(loc=2.2, scale=0.7, size=n_fraud)
    x_fraud[:, 3] = np.random.normal(loc=3.1, scale=0.7, size=n_fraud)
    x_fraud[:, 10] = np.random.normal(loc=2.6, scale=0.6, size=n_fraud)
    x_fraud[:, 11] = np.random.normal(loc=-3.4, scale=0.7, size=n_fraud)
    x_fraud[:, 13] = np.random.normal(loc=-3.8, scale=0.8, size=n_fraud)
    x_fraud[:, 16] = np.random.normal(loc=-2.9, scale=0.7, size=n_fraud)
    x_fraud[:, 28] = np.random.normal(loc=1.9, scale=0.7, size=n_fraud)

    x_all = np.vstack([x_legit, x_fraud])
    y_all = np.concatenate([
        np.zeros(n_legit, dtype=np.float32),
        np.ones(n_fraud, dtype=np.float32),
    ])

    x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(90):
        optimizer.zero_grad()
        logits = model(x_tensor).squeeze(-1)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    save_model(model, target_dir=path_obj.parent, model_name=path_obj.name)
    return model


def _generate_synthetic_creditcard_df(n_samples: int = 2000) -> pd.DataFrame:
    """Generates a realistic imbalanced creditcard + PayPal telemetry DataFrame when CSV is absent."""
    rng = np.random.default_rng(42)
    n_fraud = max(int(n_samples * 0.08), 30)
    n_legit = n_samples - n_fraud

    v_legit = rng.normal(loc=0.0, scale=0.8, size=(n_legit, 28))
    v_fraud = rng.normal(loc=0.0, scale=1.1, size=(n_fraud, 28))
    v_fraud[:, 0] -= 2.6
    v_fraud[:, 1] += 2.1
    v_fraud[:, 3] += 2.8
    v_fraud[:, 10] += 2.4
    v_fraud[:, 11] -= 3.1
    v_fraud[:, 13] -= 3.5

    features = np.vstack([v_legit, v_fraud])
    amounts = np.concatenate([
        rng.exponential(scale=65.0, size=n_legit),
        rng.exponential(scale=420.0, size=n_fraud) + 120.0,
    ])
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(features, columns=cols)
    df.insert(0, "Time", np.arange(n_samples, dtype=float))
    df["Amount"] = amounts
    df["Class"] = labels
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def run_cli_training() -> None:
    parser = argparse.ArgumentParser(description="Train Residual TxGuard Model with AdamW and OneCycleLR")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dataset_path", type=str, default="data/creditcard.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.dataset_path):
        print(f"[INFO] Loading dataset from {args.dataset_path}")
        df = pd.read_csv(args.dataset_path)
    else:
        print(
            f"[INFO] Dataset file '{args.dataset_path}' not found locally; "
            "synthesizing calibrated PayPal & card transaction dataset for training."
        )
        df = _generate_synthetic_creditcard_df(n_samples=2400)

    df["Amount"] = np.log1p(df["Amount"])
    X = df.drop(columns=["Time", "Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_loader, test_loader = create_dataloaders(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = RiskScorerModel(
        input_features=X_train_scaled.shape[1],
        hidden_units=args.hidden_units,
        dropout_rate=0.3,
    ).to(device)

    loss_fn = CostSensitiveFocalLoss(alpha=0.75, gamma=4.0, fn_weight=8.0)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.2,
    )

    train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        patience=args.patience,
        scheduler=scheduler,
    )

    save_model(model=model, target_dir=MODEL_PATH.parent, model_name=MODEL_PATH.name)

