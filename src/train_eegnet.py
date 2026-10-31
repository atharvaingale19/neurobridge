import numpy as np
import torch
import torch.nn as nn
import joblib

from prepare_data import load_eeg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")

X, y = load_eeg()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, (22, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),

            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Linear(16 * 27, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


all_predictions = np.zeros(len(y), dtype=int)
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X_train[:, None], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, None], dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=16,
        shuffle=True
    )

    model = EEGNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(40):
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.argmax(dim=1).numpy()

    all_predictions[test_idx] = predictions

    score = balanced_accuracy_score(y[test_idx], predictions)
    fold_scores.append(score)

    print(f"Fold {fold}: {score:.4f}")

print("\nMean fold balanced accuracy:", np.mean(fold_scores))
print("OOF balanced accuracy:", balanced_accuracy_score(y, all_predictions))

print("\nClassification report:")
print(classification_report(
    y,
    all_predictions,
    target_names=["Left hand", "Right hand"]
))

print("\nTraining final model...")

mean = X.mean(axis=(0, 2), keepdims=True)
std = X.std(axis=(0, 2), keepdims=True) + 1e-6

X_normalized = (X - mean) / std

X_tensor = torch.tensor(X_normalized[:, None], dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

final_model = EEGNet().to(device)

optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=16,
    shuffle=True
)

final_model.train()

for epoch in range(40):
    for inputs, labels in loader:
        optimizer.zero_grad()
        loss = criterion(final_model(inputs), labels)
        loss.backward()
        optimizer.step()

torch.save(final_model.state_dict(), "models/eegnet.pth")

joblib.dump(
    {"mean": mean, "std": std},
    "models/eegnet_normalization.joblib"
)

print("EEGNet saved successfully.")