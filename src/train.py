import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import TabularDataset
from model import TabularModel


# ======================
# CONFIG
# ======================
DATA_PATH = "data/mobile_device_usage.csv"
TARGET_COL = "Action"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)

# Nadir sınıfı birleştir
df["Action"] = df["Action"].replace({"reset-both": "deny"})


# ======================
# FEATURES & TARGET
# ======================
y_raw = df[TARGET_COL].astype(str)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

categorical_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != TARGET_COL]

X_numeric = df[numeric_cols]
X_categorical = df[categorical_cols]


# ======================
# ENCODING
# ======================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

scaler = StandardScaler()
X_num = scaler.fit_transform(X_numeric).astype(np.float32)

cat_encoders = {}
cat_cardinalities = []
X_cat_encoded_cols = []

for col in categorical_cols:
    le = LabelEncoder()
    vals = le.fit_transform(X_categorical[col].astype(str))
    cat_encoders[col] = le
    cat_cardinalities.append(len(le.classes_))
    X_cat_encoded_cols.append(vals)

if len(X_cat_encoded_cols) > 0:
    X_cat = np.vstack(X_cat_encoded_cols).T.astype(np.int64)
else:
    X_cat = np.zeros((len(df), 0), dtype=np.int64)


# ======================
# SPLIT
# ======================
(X_num_train, X_num_temp,
 X_cat_train, X_cat_temp,
 y_train, y_temp) = train_test_split(
    X_num, X_cat, y, test_size=0.3, random_state=42, stratify=y
)

(X_num_val, X_num_test,
 X_cat_val, X_cat_test,
 y_val, y_test) = train_test_split(
    X_num_temp, X_cat_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# ======================
# DATA LOADERS
# ======================
train_ds = TabularDataset(X_num_train, X_cat_train, y_train)
val_ds   = TabularDataset(X_num_val,   X_cat_val,   y_val)
test_ds  = TabularDataset(X_num_test,  X_cat_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ======================
# MODEL
# ======================
model = TabularModel(
    num_numeric_features=X_num.shape[1],
    cat_cardinalities=cat_cardinalities,
    num_classes=num_classes
).to(DEVICE)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ======================
# TRAIN / VAL
# ======================
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds_all = []
    targets_all = []

    for Xn, Xc, yb in loader:
        Xn = Xn.to(DEVICE)
        Xc = Xc.to(DEVICE)
        yb = yb.to(DEVICE)

        if train:
            optimizer.zero_grad()

        out = model(Xn, Xc)
        loss = criterion(out, yb)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(yb)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        preds_all.append(preds)
        targets_all.append(yb.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average="macro")
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, f1


best_val_f1 = 0
best_state = None

for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1 = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_f1 = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch+1:02d} | Train F1={train_f1:.3f}, Val F1={val_f1:.3f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()


# ======================
# TEST
# ======================
model.load_state_dict(best_state)

test_loss, test_acc, test_f1 = run_epoch(test_loader)
print("\n=== TEST RESULTS ===")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
print("Test F1:", test_f1)

preds = []
targets = []

with torch.no_grad():
    for Xn, Xc, yb in test_loader:
        logits = model(Xn.to(DEVICE), Xc.to(DEVICE))
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(yb.numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

print("\nClassification Report:")
print(classification_report(targets, preds, target_names=label_encoder.classes_))
