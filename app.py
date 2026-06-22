"""
EAI 6010 Module 5 - Red Wine Quality Microservice
Deploys the binary classifier (good = quality >= 7) from the Module 4
PyTorch assignment as a Gradio web service on Hugging Face Spaces.

The model architecture, preprocessing, and training procedure are identical
to the Module 4 notebook (two hidden layers of 512 units with ReLU,
StandardScaler on the training split, CrossEntropyLoss, Adam, 100 epochs).
Because the Module 4 notebook did not export a saved model file, the service
trains the model once on startup from the UCI dataset, then serves predictions.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import gradio as gr

# UCI Red Wine Quality dataset (semicolon-separated)
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


class WineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train_binary_model():
    """Reproduce the Module 4 binary model. Returns (model, scaler)."""
    df = pd.read_csv(DATA_URL, sep=";")
    X = df.drop(columns=["quality"]).values
    y_bin = (df["quality"].values >= 7).astype(int)

    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    scaler = StandardScaler().fit(Xb_tr)
    Xb_tr = scaler.transform(Xb_tr)

    train_loader = DataLoader(WineDataset(Xb_tr, yb_tr), batch_size=64, shuffle=True)

    model = NeuralNetwork(X.shape[1], 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(100):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    return model, scaler


print("Training model on startup...")
MODEL, SCALER = train_binary_model()
print("Model ready.")


def predict(
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
    free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
):
    """Take 11 physicochemical features, return a quality verdict + probability."""
    features = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,
    ]], dtype=np.float32)

    x = torch.tensor(SCALER.transform(features), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    label = "Good (quality >= 7)" if probs[1] >= 0.5 else "Not good (quality < 7)"
    return {
        "Not good (quality < 7)": float(probs[0]),
        "Good (quality >= 7)": float(probs[1]),
    }, label


# A real "good" wine from the dataset (quality = 8) and a typical "not good" one.
EXAMPLE_GOOD = [7.9, 0.35, 0.46, 3.6, 0.078, 15, 37, 0.9973, 3.35, 0.86, 12.8]
EXAMPLE_NOT_GOOD = [7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]

DESCRIPTION = """
This service exposes the **binary red wine classifier** built in the EAI 6010
Module 4 assignment. It takes the 11 physicochemical measurements of a red wine
sample and predicts whether the wine is **good** (sensory quality score of 7 or
higher) or **not good** (score below 7), along with the model's probability for
each class.

**Model limitations (please read):** The classifier reaches roughly 91% accuracy
on the held-out test set, but that figure is inflated because only ~14% of wines
are "good"; a trivial model predicting "not good" for everything would already
reach ~86%. Recall on the "good" class is about 0.65, so the model misses roughly
one in three high-quality wines. It is a course demonstration, not a
production-grade quality grader.
"""

inputs = [
    gr.Number(label="Fixed acidity (g/dm^3)", value=EXAMPLE_NOT_GOOD[0]),
    gr.Number(label="Volatile acidity (g/dm^3)", value=EXAMPLE_NOT_GOOD[1]),
    gr.Number(label="Citric acid (g/dm^3)", value=EXAMPLE_NOT_GOOD[2]),
    gr.Number(label="Residual sugar (g/dm^3)", value=EXAMPLE_NOT_GOOD[3]),
    gr.Number(label="Chlorides (g/dm^3)", value=EXAMPLE_NOT_GOOD[4]),
    gr.Number(label="Free sulfur dioxide (mg/dm^3)", value=EXAMPLE_NOT_GOOD[5]),
    gr.Number(label="Total sulfur dioxide (mg/dm^3)", value=EXAMPLE_NOT_GOOD[6]),
    gr.Number(label="Density (g/cm^3)", value=EXAMPLE_NOT_GOOD[7]),
    gr.Number(label="pH", value=EXAMPLE_NOT_GOOD[8]),
    gr.Number(label="Sulphates (g/dm^3)", value=EXAMPLE_NOT_GOOD[9]),
    gr.Number(label="Alcohol (% vol)", value=EXAMPLE_NOT_GOOD[10]),
]

outputs = [
    gr.Label(label="Class probabilities"),
    gr.Textbox(label="Prediction"),
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Red Wine Quality Classifier (EAI 6010 Module 5)",
    description=DESCRIPTION,
    examples=[EXAMPLE_NOT_GOOD, EXAMPLE_GOOD],
    cache_examples=False,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)
