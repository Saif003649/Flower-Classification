"""
Flower Classification — Model Test & Verification Script
Supports both:
1. Custom CNN (PyTorch baseline from model/flower_classification_model.pth)
2. Vision Transformer (ViT-Base from loretyan/vit-base-oxford-flowers-102)

Usage:
  python test_model.py               # Tests both models
  python test_model.py --model cnn   # Tests only Custom CNN
  python test_model.py --model vit   # Tests only Vision Transformer
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "flower_classification_model.pth")
VIT_MODEL_ID   = "loretyan/vit-base-oxford-flowers-102"
IMG_DIR        = os.path.join(BASE_DIR, "static", "uploads")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIVIDER        = "=" * 70

# ── Canonical Oxford 102 Class Names (for CNN) ──────────────────────────────
CNN_CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barberton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia",
    "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]

# ── CNN Architecture ────────────────────────────────────────────────────────
class FlowerCNN(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def clean_label(name: str) -> str:
    n = str(name).strip().rstrip("?")
    if n.lower() == "barbeton daisy":
        return "barberton daisy"
    return n

def test_cnn(images):
    print(f"\n{DIVIDER}")
    print("  EVALUATING MODEL 1: CUSTOM CNN (PyTorch)")
    print(f"{DIVIDER}")

    if not os.path.exists(CNN_MODEL_PATH):
        print(f"[-] Checkpoint not found: {CNN_MODEL_PATH}")
        return

    model = FlowerCNN(num_classes=len(CNN_CLASS_NAMES))
    state = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded: {os.path.basename(CNN_MODEL_PATH)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for idx, fname in enumerate(images, 1):
        img_path = os.path.join(IMG_DIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = cnn_transform(img).unsqueeze(0).to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            lat_ms = (time.time() - t0) * 1000

            top3_probs, top3_idx = torch.topk(probs, k=3)
            print(f"\n  Image [{idx}]: {fname} (Inference: {lat_ms:.1f} ms)")
            for r, (p, i) in enumerate(zip(top3_probs, top3_idx), 1):
                lbl = CNN_CLASS_NAMES[i.item()].title()
                print(f"    #{r} {lbl:<26} ({p.item()*100:.2f}%)")
        except Exception as e:
            print(f"  [-] Error on {fname}: {e}")

def test_vit(images):
    print(f"\n{DIVIDER}")
    print("  EVALUATING MODEL 2: VISION TRANSFORMER (ViT-Base)")
    print(f"{DIVIDER}")

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        print("[-] transformers is not installed. Run: pip install transformers")
        return

    try:
        processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID)
        model.to(DEVICE)
        model.eval()

        vit_id2label = {int(k): clean_label(v) for k, v in model.config.id2label.items()}
        print(f"Loaded: {VIT_MODEL_ID}")
        print(f"Classes: {len(vit_id2label)}")
    except Exception as e:
        print(f"[-] Error loading ViT model: {e}")
        return

    for idx, fname in enumerate(images, 1):
        img_path = os.path.join(IMG_DIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)

            t0 = time.time()
            with torch.inference_mode():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            lat_ms = (time.time() - t0) * 1000

            top3_probs, top3_idx = torch.topk(probs, k=3)
            print(f"\n  Image [{idx}]: {fname} (Inference: {lat_ms:.1f} ms)")
            for r, (p, i) in enumerate(zip(top3_probs, top3_idx), 1):
                lbl = vit_id2label.get(i.item(), f"Class {i.item()}").title()
                print(f"    #{r} {lbl:<26} ({p.item()*100:.2f}%)")
        except Exception as e:
            print(f"  [-] Error on {fname}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Flower Classification Models")
    parser.add_argument("--model", choices=["both", "cnn", "vit"], default="both",
                        help="Which model to test (default: both)")
    args = parser.parse_args()

    print(f"\n{DIVIDER}")
    print("  FLOWER CLASSIFICATION — MODEL TEST RUNNER")
    print(f"{DIVIDER}")
    print(f"Device: {DEVICE}")

    valid_exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in sorted(os.listdir(IMG_DIR)) if os.path.splitext(f)[1].lower() in valid_exts]
    if not images:
        print(f"[-] No test images found in {IMG_DIR}")
        return

    print(f"Found {len(images)} test image(s) in static/uploads/")

    if args.model in ("both", "cnn"):
        test_cnn(images)

    if args.model in ("both", "vit"):
        test_vit(images)

    print(f"\n{DIVIDER}")
    print("  Test run completed successfully.")
    print(f"{DIVIDER}\n")

if __name__ == "__main__":
    main()
