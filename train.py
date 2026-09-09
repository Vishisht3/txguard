import argparse
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import data_setup, model_builder, engine, utils

def parse_args():
    parser = argparse.ArgumentParser(description="Train Regularized TxGuard Model")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/creditcard.csv"
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path non-existent: {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_loader, test_loader = data_setup.create_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test, batch_size=args.batch_size
    )

    model = model_builder.RiskScorerModel(
        input_features=X_train_scaled.shape[1], 
        hidden_units=args.hidden_units,
        dropout_rate=0.3
    ).to(device)

    num_positives = (y_train == 1).sum()
    num_negatives = (y_train == 0).sum()
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    engine.train(
        model=model, 
        train_dataloader=train_loader, 
        test_dataloader=test_loader, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        epochs=args.epochs, 
        device=device,
        patience=args.patience
    )

    MODEL_SAVE_DIR = "models"
    utils.save_model(model=model, target_dir=MODEL_SAVE_DIR, model_name="txguard_model.pth")

if __name__ == "__main__":
    main()
