"""
Kealzyl Neuron
A small phrase classifier with a Flask web interface and a PyTorch neural network.

Run:
    py -m pip install -r requirements.txt
    py kealzyl_neuron.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, render_template, request
from torch import nn

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "kealzyl_neuron.pth"
OPTIMIZER_PATH = APP_DIR / "kealzyl_neuron_optimizer.pth"
DATA_PATH = APP_DIR / "training_data.json"

LABELS = ["yes", "no", "maybe", "surely not", "of course"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
FEATURE_COUNT = 10
HIDDEN_SIZE = 10
HIDDEN_LAYERS = 10

# Small, editable seed dataset. Add examples through the web interface.
SEED_DATA = [
    ("yes", "yes"), ("yes", "yeah"), ("yes", "yep"), ("yes", "absolutely"),
    ("yes", "I agree"), ("yes", "that is correct"), ("yes", "yes, please"),
    ("yes", "definitely yes"), ("yes", "sounds good"), ("yes", "I think so"),
    ("no", "no"), ("no", "nope"), ("no", "I disagree"), ("no", "not at all"),
    ("no", "I don't think so"), ("no", "never"), ("no", "that is incorrect"),
    ("no", "absolutely not"), ("no", "I cannot"), ("no", "not really"),
    ("maybe", "maybe"), ("maybe", "perhaps"), ("maybe", "I am not sure"),
    ("maybe", "possibly"), ("maybe", "it depends"), ("maybe", "could be"),
    ("maybe", "I might"), ("maybe", "not certain"), ("maybe", "we will see"),
    ("maybe", "I guess so"), ("surely not", "surely not"),
    ("surely not", "definitely not"), ("surely not", "no way"),
    ("surely not", "absolutely not"), ("surely not", "certainly not"),
    ("surely not", "impossible"), ("surely not", "I would never"),
    ("surely not", "there is no chance"), ("surely not", "not a chance"),
    ("surely not", "of course not"), ("of course", "of course"),
    ("of course", "of course yes"), ("of course", "certainly"),
    ("of course", "naturally"), ("of course", "without a doubt"),
    ("of course", "sure thing"), ("of course", "you bet"),
    ("of course", "for sure"), ("of course", "obviously"),
    ("of course", "most definitely"),
]

# Features are intentionally transparent and lightweight, not an LLM embedding.
POSITIVE_WORDS = {
    "yes", "yeah", "yep", "agree", "correct", "absolutely", "definitely",
    "sure", "certainly", "naturally", "good", "right", "please", "bet",
    "course", "okay", "ok", "true", "exactly", "fine"
}
NEGATIVE_WORDS = {
    "no", "nope", "not", "never", "cannot", "can't", "won't", "wrong",
    "disagree", "impossible", "neither", "deny", "refuse", "without"
}
UNCERTAIN_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "depends", "guess",
    "unsure", "uncertain", "uncertain", "think", "probably", "possibly",
    "wonder", "doubt"
}
STRONG_WORDS = {
    "absolutely", "definitely", "certainly", "surely", "never", "impossible",
    "undoubtedly", "always", "obviously", "completely", "totally", "without"
}
QUESTION_WORDS = {"what", "why", "when", "where", "who", "how", "can", "could", "would", "will", "is", "are", "do"}
NEGATIONS = {"not", "never", "no", "cannot", "can't", "won't", "isn't", "don't", "doesn't"}

def extract_features(text: str) -> list[float]:
    """Convert a phrase to 10 normalized numeric features."""
    cleaned = text.lower().strip()
    words = re.findall(r"[a-z']+", cleaned)
    count = max(len(words), 1)
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    uncertain = sum(w in UNCERTAIN_WORDS for w in words)
    strong = sum(w in STRONG_WORDS for w in words)
    questions = sum(w in QUESTION_WORDS for w in words)
    negation = sum(w in NEGATIONS for w in words)
    exclamations = cleaned.count("!")
    question_marks = cleaned.count("?")
    length = min(len(words), 40)

    # Keep each feature in a modest numeric range.
    return [
        min(pos / count, 1.0),
        min(neg / count, 1.0),
        min(uncertain / count, 1.0),
        min(strong / count, 1.0),
        min(questions / count, 1.0),
        min(negation / count, 1.0),
        min(exclamations, 3) / 3.0,
        min(question_marks, 3) / 3.0,
        length / 40.0,
        1.0 if any(w in cleaned for w in (" of course", "surely not", "no way", "not a chance")) else 0.0,
    ]


class KealzylNeuron(nn.Module):
    """10 hidden layers, each with 10 neurons, followed by 5 class logits."""
    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(FEATURE_COUNT, HIDDEN_SIZE))
        layers.append(nn.ReLU())
        for _ in range(HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(HIDDEN_SIZE, len(LABELS))

    def forward(self, x):
        return self.output(self.hidden(x))


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KealzylNeuron().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()
model_lock = threading.Lock()
training_in_progress = False
last_training_info: dict[str, Any] = {"status": "Not trained in this session yet."}


def load_training_data() -> list[dict[str, str]]:
    if not DATA_PATH.exists():
        rows = [{"text": text, "label": label} for label, text in SEED_DATA]
        save_training_data(rows)
        return rows
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        return [
            {"text": str(row["text"]), "label": str(row["label"])}
            for row in data
            if isinstance(row, dict) and row.get("label") in LABEL_TO_ID and str(row.get("text", "")).strip()
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return [{"text": text, "label": label} for label, text in SEED_DATA]


def save_training_data(rows: list[dict[str, str]]) -> None:
    DATA_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


training_data = load_training_data()


def make_tensors(rows: list[dict[str, str]]):
    x = torch.tensor([extract_features(row["text"]) for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([LABEL_TO_ID[row["label"]] for row in rows], dtype=torch.long, device=device)
    return x, y


def safe_torch_load(path: Path):
    # weights_only is supported in current PyTorch versions. Fall back for older versions.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint() -> None:
    with model_lock:
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(optimizer.state_dict(), OPTIMIZER_PATH)


def load_checkpoint() -> bool:
    if not MODEL_PATH.exists():
        return False
    with model_lock:
        state = safe_torch_load(MODEL_PATH)
        model.load_state_dict(state)
        if OPTIMIZER_PATH.exists():
            try:
                optimizer.load_state_dict(safe_torch_load(OPTIMIZER_PATH))
            except (ValueError, RuntimeError, KeyError):
                pass
    return True


def train_model(epochs: int = 120) -> dict[str, Any]:
    global training_in_progress, last_training_info
    if training_in_progress:
        raise RuntimeError("Training is already running.")
    rows = list(training_data)
    if len(rows) < 5:
        raise ValueError("Add at least 5 training examples first.")

    training_in_progress = True
    try:
        x, y = make_tensors(rows)
        epochs = max(1, min(int(epochs), 1000))
        with model_lock:
            model.train()
            final_loss = 0.0
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())
            model.eval()
        save_checkpoint()
        last_training_info = {
            "status": "Training completed",
            "examples": len(rows),
            "epochs": epochs,
            "loss": round(final_loss, 6),
        }
        return last_training_info
    finally:
        training_in_progress = False


# Load saved weights if present. Otherwise the web UI will offer training.
try:
    checkpoint_loaded = load_checkpoint()
except (RuntimeError, ValueError, OSError):
    checkpoint_loaded = False


@app.get("/")
def index():
    return render_template(
        "index.html",
        labels=LABELS,
        device_name=str(device),
        model_loaded=checkpoint_loaded,
        example_count=len(training_data),
        hidden_layers=HIDDEN_LAYERS,
        hidden_size=HIDDEN_SIZE,
    )


@app.get("/api/status")
def api_status():
    return jsonify({
        "labels": LABELS,
        "device": str(device),
        "examples": len(training_data),
        "model_saved": MODEL_PATH.exists(),
        "optimizer_saved": OPTIMIZER_PATH.exists(),
        "training": training_in_progress,
        "last_training": last_training_info,
    })


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    phrase = str(body.get("text", "")).strip()
    if not phrase:
        return jsonify({"error": "Type a phrase first."}), 400
    if len(phrase) > 1000:
        return jsonify({"error": "Please keep phrases under 1,000 characters."}), 400

    tensor = torch.tensor([extract_features(phrase)], dtype=torch.float32, device=device)
    with model_lock:
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    best = max(range(len(LABELS)), key=lambda i: probabilities[i])
    return jsonify({
        "phrase": phrase,
        "prediction": LABELS[best],
        "confidence": round(probabilities[best] * 100, 2),
        "probabilities": [
            {"label": LABELS[i], "percent": round(probabilities[i] * 100, 2)}
            for i in range(len(LABELS))
        ],
        "trained": MODEL_PATH.exists(),
        "note": "Confidence is the network's score, not a guarantee of correctness."
    })


@app.post("/api/examples")
def api_add_example():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    label = str(body.get("label", "")).strip().lower()
    if not text:
        return jsonify({"error": "Enter a training phrase."}), 400
    if len(text) > 1000:
        return jsonify({"error": "Please keep phrases under 1,000 characters."}), 400
    if label not in LABEL_TO_ID:
        return jsonify({"error": "Choose one of the five valid labels."}), 400
    training_data.append({"text": text, "label": label})
    save_training_data(training_data)
    return jsonify({"message": "Training example added.", "examples": len(training_data)})



@app.post("/api/examples/bulk")
def api_add_examples_bulk():
    """Add many examples in one request. Each row is {text, label}."""
    body = request.get_json(silent=True) or {}
    rows = body.get("examples", [])
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Paste at least one training example."}), 400
    if len(rows) > 5000:
        return jsonify({"error": "Please add no more than 5,000 examples at once."}), 400

    accepted = []
    errors = []
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"Row {line_number}: invalid row.")
            continue
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not text:
            errors.append(f"Row {line_number}: phrase is empty.")
            continue
        if len(text) > 1000:
            errors.append(f"Row {line_number}: phrase is over 1,000 characters.")
            continue
        if label not in LABEL_TO_ID:
            errors.append(f"Row {line_number}: unknown label '{label}'.")
            continue
        accepted.append({"text": text, "label": label})

    if not accepted:
        return jsonify({"error": "No valid examples found.", "errors": errors[:12]}), 400

    training_data.extend(accepted)
    save_training_data(training_data)
    return jsonify({
        "message": f"Added {len(accepted)} training examples.",
        "added": len(accepted),
        "rejected": len(errors),
        "errors": errors[:12],
        "examples": len(training_data),
    })


@app.post("/api/train")
def api_train():
    body = request.get_json(silent=True) or {}
    try:
        result = train_model(body.get("epochs", 120))
        return jsonify(result)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Training failed")
        return jsonify({"error": f"Training failed: {exc}"}), 500


@app.post("/api/save")
def api_save():
    try:
        save_checkpoint()
        return jsonify({
            "message": "Model weights and optimizer state saved.",
            "model_file": MODEL_PATH.name,
            "optimizer_file": OPTIMIZER_PATH.name,
        })
    except Exception as exc:
        app.logger.exception("Save failed")
        return jsonify({"error": f"Could not save model: {exc}"}), 500


@app.post("/api/load")
def api_load():
    global checkpoint_loaded
    try:
        if not MODEL_PATH.exists():
            return jsonify({"error": "No saved model found yet. Train the network first."}), 404
        checkpoint_loaded = load_checkpoint()
        return jsonify({"message": "Saved model loaded."})
    except Exception as exc:
        app.logger.exception("Load failed")
        return jsonify({"error": f"Could not load model: {exc}"}), 500


@app.get("/api/examples")
def api_examples():
    # Return only the newest 100 examples to keep the browser response small.
    return jsonify({"examples": list(reversed(training_data[-100:])), "total": len(training_data)})


if __name__ == "__main__":
    print("Kealzyl Neuron is starting.")
    print(f"Device: {device}")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
