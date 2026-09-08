import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import data_setup, model_builder, engine, utils

parser = argparse.ArgumentParser(description="Train TxGuard AI Model")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/creditcard.csv"
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

model = model_builder.RiskScorerModel(input_features=X_train_scaled.shape[1], hidden_units=args.hidden_units).to(device)

num_positives = (y_train == 1).sum()
num_negatives = (y_train == 0).sum()
pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

engine.train(model=model, train_dataloader=train_loader, test_dataloader=test_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=args.epochs, device=device)

MODEL_SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/txguard/models"
utils.save_model(model=model, target_dir=MODEL_SAVE_DIR, model_name="txguard_model.pth")
