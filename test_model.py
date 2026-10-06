"""
Flower Classification Model — Complete Test & Evaluation Script
Runs inference on all images found in static/uploads/
Also shows: top-5 predictions, confidence for each image,
model summary, parameter count, and per-class probabilities.
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "flower_classification_model.pth")
IMG_DIR    = os.path.join(BASE_DIR, "static", "uploads")

# ── Class Labels (102 Oxford Flowers) ───────────────────────────────────────
CLASS_NAMES = [
    "pink primrose","hard-leaved pocket orchid","canterbury bells",
    "sweet pea","english marigold","tiger lily","moon orchid",
    "bird of paradise","monkshood","globe thistle","snapdragon",
    "colt's foot","king protea","spear thistle","yellow iris",
    "globe-flower","purple coneflower","peruvian lily","balloon flower",
    "giant white arum lily","fire lily","pincushion flower","fritillary",
    "red ginger","grape hyacinth","corn poppy","prince of wales feathers",
    "stemless gentian","artichoke","sweet william","carnation",
    "garden phlox","love in the mist","mexican aster","alpine sea holly",
    "ruby-lipped cattleya","cape flower","great masterwort","siam tulip",
    "lenten rose","barberton daisy","daffodil","sword lily",
    "poinsettia","bolero deep blue","wallflower","marigold",
    "buttercup","oxeye daisy","common dandelion","petunia",
    "wild pansy","primula","sunflower","pelargonium",
    "bishop of llandaff","gaura","geranium","orange dahlia",
    "pink-yellow dahlia","cautleya spicata","japanese anemone",
    "black-eyed susan","silverbush","californian poppy","osteospermum",
    "spring crocus","iris","windflower","tree poppy","gazania",
    "azalea","water lily","rose","thorn apple","morning glory",
    "passion flower","lotus","toad lily","anthurium","frangipani",
    "clematis","hibiscus","columbine","desert-rose","tree mallow",
    "magnolia","cyclamen","watercress","canna lily","hippeastrum",
    "bee balm","ball moss","foxglove","bougainvillea","camellia",
    "mallow","mexican petunia","bromelia","blanket flower",
    "trumpet creeper","blackberry lily",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Model Architecture ───────────────────────────────────────────────────────
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

# ── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DIVIDER = "=" * 65

# ── 1. Model Info ────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  FLOWER CLASSIFICATION MODEL — TEST REPORT")
print(f"{DIVIDER}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device        : {device}")
print(f"  Model file    : {os.path.basename(MODEL_PATH)}")
print(f"  Model size    : {os.path.getsize(MODEL_PATH) / 1e6:.2f} MB")
print(f"  Output classes: {NUM_CLASSES}")
print(f"  Input size    : 224 × 224")

# ── 2. Load Model ────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  LOADING MODEL")
print(DIVIDER)

model = FlowerCNN(NUM_CLASSES)
t0    = time.time()
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
load_time = time.time() - t0

total_params   = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  ✅ Model loaded in  : {load_time:.3f} s")
print(f"  Total parameters    : {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ── 3. Layer Summary ─────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  LAYER SUMMARY")
print(DIVIDER)
print(f"\n  {'Layer':<35} {'Output Shape':<22} {'Params':>10}")
print(f"  {'-'*35} {'-'*22} {'-'*10}")

layer_info = [
    ("Conv2d-1  (3→16, k=3, p=1)",   "[B, 16, 224, 224]",  16*3*3*3 + 16),
    ("ReLU-1",                         "[B, 16, 224, 224]",  0),
    ("MaxPool2d-1  (2×2)",             "[B, 16, 112, 112]",  0),
    ("Conv2d-2  (16→32, k=3, p=1)",   "[B, 32, 112, 112]",  32*16*3*3 + 32),
    ("ReLU-2",                         "[B, 32, 112, 112]",  0),
    ("MaxPool2d-2  (2×2)",             "[B, 32,  56,  56]",  0),
    ("Flatten",                        "[B, 100352]",         0),
    ("Linear  (100352 → 128)",         "[B, 128]",            100352*128 + 128),
    ("ReLU-3",                         "[B, 128]",            0),
    ("Linear  (128 → 102)",            "[B, 102]",            128*102 + 102),
]
for name, shape, params in layer_info:
    print(f"  {name:<35} {shape:<22} {params:>10,}")
print(f"  {'─'*35} {'─'*22} {'─'*10}")
print(f"  {'TOTAL':<35} {'':22} {total_params:>10,}")

# ── 4. Inference on Uploaded Images ─────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  INFERENCE ON TEST IMAGES")
print(DIVIDER)

valid_exts = {".jpg", ".jpeg", ".png"}
images = [
    f for f in os.listdir(IMG_DIR)
    if os.path.splitext(f)[1].lower() in valid_exts
]

if not images:
    print("\n  ⚠  No images found in static/uploads/")
    print("     Upload some images via the web app first, then re-run.\n")
    sys.exit(0)

print(f"\n  Found {len(images)} image(s) to test.\n")

results = []
for idx, fname in enumerate(images, 1):
    img_path = os.path.join(IMG_DIR, fname)
    try:
        img    = Image.open(img_path).convert("RGB")
        w, h   = img.size
        tensor = transform(img).unsqueeze(0).to(device)

        t_start = time.time()
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
        infer_ms = (time.time() - t_start) * 1000

        top5_probs, top5_idx = torch.topk(probs, 5, dim=1)
        top5 = [(CLASS_NAMES[i], round(p * 100, 2))
                for i, p in zip(top5_idx[0].tolist(), top5_probs[0].tolist())]

        pred_class  = top5[0][0]
        pred_conf   = top5[0][1]
        results.append((fname, pred_class, pred_conf))

        print(f"  ── Image {idx}: {fname}")
        print(f"     Size         : {w} × {h} px")
        print(f"     Inference    : {infer_ms:.1f} ms")
        print(f"     ✅ Prediction : {pred_class.upper()}  ({pred_conf}%)")
        print(f"     Top-5 Results:")
        for rank, (cls, conf) in enumerate(top5, 1):
            bar = "█" * int(conf / 5)
            print(f"       {rank}. {cls:<30}  {conf:6.2f}%  {bar}")
        print()

    except Exception as e:
        print(f"  ❌ Error on {fname}: {e}\n")

# ── 5. Summary Table ─────────────────────────────────────────────────────────
print(f"{DIVIDER}")
print("  SUMMARY")
print(DIVIDER)
print(f"\n  {'#':<4} {'Image File':<44} {'Predicted Class':<28} {'Confidence':>10}")
print(f"  {'─'*4} {'─'*44} {'─'*28} {'─'*10}")
for i, (fname, cls, conf) in enumerate(results, 1):
    short = fname[:42] + ".." if len(fname) > 44 else fname
    print(f"  {i:<4} {short:<44} {cls.title():<28} {conf:>9.2f}%")

print(f"\n{DIVIDER}")
print("  NOTE: No labelled test dataset was provided.")
print("  Accuracy cannot be computed without ground-truth labels.")
print("  The predictions above show raw model output on uploaded images.")
print(f"{DIVIDER}\n")
