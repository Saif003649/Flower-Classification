"""
Benchmark Script: Custom CNN vs Vision Transformer (ViT-Base)
=============================================================
Evaluates both models on the official Oxford Flowers 102 test split (6,149 images).

Metrics reported:
  - Top-1 Accuracy, Top-5 Accuracy
  - Macro-Precision, Macro-Recall, Macro-F1
  - Per-image steady-state inference time (model forward pass only)
  - Model loading time (separate from inference timing)
  - Checkpoint / weight file size
  - Peak RSS memory usage (via ctypes if psutil unavailable)

Output files:
  benchmark_results.json       -- Full metrics dict with metadata
  benchmark_comparison.csv     -- Side-by-side tabular comparison

Usage:
  python benchmark_models.py                 # Evaluates both models (ViT ~22 min on CPU)
  python benchmark_models.py --skip-vit-eval # Load ViT metrics from existing JSON (fast)

Constraints:
  - Does NOT modify app.py, existing checkpoints, templates, or static files.
  - Does NOT train or tune on the test set.
  - Timing EXCLUDES image loading; includes only the model's forward pass.
  - Model loading time is measured and reported separately.

Label alignment:
  - CNN: CLASS_NAMES[argmax] matches Flowers102.classes canonical order directly.
  - ViT: model.config.id2label has known 5-class cyclic rotation at indices 16-20
          and minor typos; resolved via name-to-canonical-index mapping.
"""

import os
import sys
import json
import time
import argparse
import tracemalloc
import csv

# ---------------------------------------------------------------------------
#  Dependency Checks
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    from PIL import Image
except ModuleNotFoundError as e:
    print(f"[ERROR] Missing required library: {e.name}")
    print("  Install: pip install torch torchvision pillow")
    sys.exit(1)

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    _TRANSFORMERS_OK = True
except ModuleNotFoundError:
    _TRANSFORMERS_OK = False

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    _SKLEARN_OK = True
except ModuleNotFoundError:
    _SKLEARN_OK = False
    print("[WARN] scikit-learn not found. F1/Precision/Recall will be skipped.")

# ---------------------------------------------------------------------------
#  Constants & Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "flower_classification_model.pth")
VIT_MODEL_ID   = "loretyan/vit-base-oxford-flowers-102"
VIT_METRICS_PATH  = os.path.join(BASE_DIR, "oxford_vit_evaluation_metrics.json")
OUT_JSON       = os.path.join(BASE_DIR, "benchmark_results.json")
OUT_CSV        = os.path.join(BASE_DIR, "benchmark_comparison.csv")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16    # Smaller than evaluate_oxford_vit.py to reduce peak memory
DIVIDER    = "=" * 72

# ─────────────────────────────────────────────────────────────────────────────
#  CLASS_NAMES (102 Oxford Flower categories — canonical order from app.py)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
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
NUM_CLASSES = len(CLASS_NAMES)  # 102

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CNN Architecture (must match flower_classification_model.pth)
# ─────────────────────────────────────────────────────────────────────────────
class FlowerCNN(nn.Module):
    def __init__(self, num_classes: int = 102):
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


# CNN preprocessing — ImageNet normalization (matches training in app.py)
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
#  Memory Measurement Helper (no psutil required)
# ─────────────────────────────────────────────────────────────────────────────
def _get_process_rss_mb() -> float:
    """Return current process RSS in MB using Windows PROCESS_MEMORY_INFO."""
    try:
        import ctypes
        import ctypes.wintypes
        # PROCESS_MEMORY_COUNTERS_EX structure
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb",                          ctypes.wintypes.DWORD),
                ("PageFaultCount",              ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize",          ctypes.c_size_t),
                ("WorkingSetSize",              ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage",     ctypes.c_size_t),
                ("QuotaPagedPoolUsage",         ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage",  ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage",      ctypes.c_size_t),
                ("PagefileUsage",               ctypes.c_size_t),
                ("PeakPagefileUsage",           ctypes.c_size_t),
                ("PrivateUsage",                ctypes.c_size_t),
            ]
        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        handle = GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / 1024 / 1024
    except Exception:
        return -1.0  # Unavailable


def _file_size_mb(path: str) -> float:
    """Return file size in MB, or -1 if path does not exist."""
    try:
        return round(os.path.getsize(path) / 1024 / 1024, 2)
    except OSError:
        return -1.0

# ─────────────────────────────────────────────────────────────────────────────
#  ViT Label Cleaning (must match app.py's clean_label_name)
# ─────────────────────────────────────────────────────────────────────────────
def clean_vit_label(raw_name: str) -> str:
    """Normalize a raw id2label value to a clean lowercase flower name."""
    name = str(raw_name).strip().rstrip("?")
    if name.lower() == "barbeton daisy":
        return "barberton daisy"
    return name.lower().strip()


def build_vit_to_canonical_map(vit_id2label: dict, canonical_classes: list) -> dict:
    """
    Build a mapping: ViT output neuron index -> canonical Flowers102 class index.
    Uses name matching to handle the known 5-class cyclic rotation at indices 16-20
    and minor label discrepancies (typos, trailing '?').
    """
    canonical_name_to_idx = {
        name.lower().strip(): i for i, name in enumerate(canonical_classes)
    }
    mapping = {}
    for vit_idx, raw_name in vit_id2label.items():
        vit_idx = int(vit_idx)
        clean = clean_vit_label(raw_name)
        if clean in canonical_name_to_idx:
            mapping[vit_idx] = canonical_name_to_idx[clean]
        elif "iris" in clean:
            # Handle "bearded iris" -> "iris" fallback
            mapping[vit_idx] = canonical_name_to_idx.get(
                "iris", canonical_name_to_idx.get("bearded iris", vit_idx)
            )
        else:
            # Fallback: keep raw index (should not happen for this model)
            mapping[vit_idx] = vit_idx
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_cnn() -> tuple:
    """Load FlowerCNN from checkpoint. Returns (model, load_time_sec, size_mb)."""
    size_mb = _file_size_mb(CNN_MODEL_PATH)
    t0 = time.perf_counter()

    m = FlowerCNN(num_classes=NUM_CLASSES)
    state = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True)

    if isinstance(state, dict) and "model_state_dict" in state:
        m.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        m.load_state_dict(state["state_dict"])
    else:
        m.load_state_dict(state)

    m.to(DEVICE)
    m.eval()
    load_time = time.perf_counter() - t0

    # Count parameters
    n_params = sum(p.numel() for p in m.parameters())
    print(f"    [OK] CNN loaded in {load_time:.2f}s | "
          f"{size_mb} MB on disk | {n_params:,} parameters")
    return m, load_time, size_mb


def load_vit() -> tuple:
    """Load ViT model and processor. Returns (processor, model, id2label, load_time_sec, size_mb_approx)."""
    if not _TRANSFORMERS_OK:
        raise RuntimeError("transformers library is not installed.")

    # Try to find the safetensors file in HF cache for size reporting
    import glob
    cache_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    sts_hits = glob.glob(os.path.join(cache_root, "**", "model.safetensors"), recursive=True)
    bin_hits = glob.glob(os.path.join(cache_root, "**", "pytorch_model.bin"), recursive=True)
    vit_files = [p for p in sts_hits + bin_hits if "loretyan" in p or "oxford" in p.lower()]
    size_mb = _file_size_mb(vit_files[0]) if vit_files else -1.0

    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    load_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    raw_id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"    [OK] ViT loaded in {load_time:.2f}s | "
          f"{size_mb} MB on disk | {n_params:,} parameters")
    return processor, model, raw_id2label, load_time, size_mb


# ─────────────────────────────────────────────────────────────────────────────
#  CNN Evaluation on Flowers102 Test Split
# ─────────────────────────────────────────────────────────────────────────────
def cnn_collate_fn(batch):
    """Apply CNN transform to PIL images; return (tensor_batch, label_tensor)."""
    images, labels = zip(*batch)
    tensors = torch.stack([CNN_TRANSFORM(img.convert("RGB")) for img in images])
    return tensors, torch.tensor(labels, dtype=torch.long)


def evaluate_cnn(cnn_model) -> dict:
    """
    Evaluate CNN on the official Oxford Flowers 102 test split.
    Timing covers ONLY the forward pass (excludes image load + preprocessing).
    """
    print("\n[*] Loading Oxford Flowers 102 test split for CNN evaluation...")
    os.makedirs(DATA_DIR, exist_ok=True)
    test_ds = datasets.Flowers102(root=DATA_DIR, split="test", download=True)
    total = len(test_ds)
    print(f"    Test images: {total:,}")

    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=cnn_collate_fn,
    )

    all_targets   = []
    all_preds     = []
    top5_hits     = 0
    inference_times_ms = []   # per-image forward-pass time (ms)

    # Memory snapshot before inference
    rss_before_mb = _get_process_rss_mb()
    tracemalloc.start()

    print(f"[*] Running CNN inference on {total:,} images "
          f"(batch={BATCH_SIZE}, device={DEVICE}) ...")

    with torch.inference_mode():
        for batch_idx, (imgs, targets) in enumerate(loader):
            imgs    = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            # --- Timed forward pass only ---
            t0 = time.perf_counter()
            logits = cnn_model(imgs)
            t1 = time.perf_counter()
            # --------------------------------

            batch_ms = (t1 - t0) * 1000.0
            per_img_ms = batch_ms / imgs.size(0)
            inference_times_ms.append(per_img_ms)

            preds = torch.argmax(logits, dim=1)
            _, top5 = torch.topk(logits, k=min(5, NUM_CLASSES), dim=1)

            top5_hits += torch.any(top5 == targets.unsqueeze(1), dim=1).sum().item()
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            processed = (batch_idx + 1) * BATCH_SIZE
            if (batch_idx + 1) % 50 == 0 or processed >= total:
                print(f"    Processed {min(processed, total):>5}/{total} images "
                      f"({min(processed, total)/total*100:>5.1f}%) "
                      f"| last batch {batch_ms:.1f}ms")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_mb = _get_process_rss_mb()

    # --- Compute metrics ---
    top1_correct = sum(p == t for p, t in zip(all_preds, all_targets))
    top1_acc = top1_correct / total * 100
    top5_acc = top5_hits / total * 100
    avg_ms   = sum(inference_times_ms) / len(inference_times_ms)
    med_ms   = sorted(inference_times_ms)[len(inference_times_ms) // 2]

    metrics = {
        "total_test_images": total,
        "top1_accuracy":     round(top1_acc, 4),
        "top5_accuracy":     round(top5_acc, 4),
        "avg_inference_ms_per_image": round(avg_ms, 4),
        "median_inference_ms_per_image": round(med_ms, 4),
        "rss_before_inference_mb": round(rss_before_mb, 2),
        "rss_after_inference_mb":  round(rss_after_mb, 2),
        "tracemalloc_peak_mb":     round(peak_mem / 1024 / 1024, 2),
    }

    if _SKLEARN_OK:
        metrics["macro_precision"] = round(
            precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100, 4)
        metrics["macro_recall"] = round(
            recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100, 4)
        metrics["macro_f1"] = round(
            f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100, 4)

    # Store raw predictions for confusion matrix (optional)
    metrics["_raw_targets"] = all_targets
    metrics["_raw_preds"]   = all_preds

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  ViT Evaluation (full re-run, or load from existing metrics JSON)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_vit_from_file() -> dict:
    """Load ViT metrics from the pre-computed evaluate_oxford_vit.py output."""
    with open(VIT_METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    sem = data.get("semantic_aligned_metrics", {})
    raw = data.get("raw_metrics", {})
    return {
        "total_test_images":             data.get("total_test_samples", 6149),
        "top1_accuracy":                 sem.get("top1_accuracy", raw.get("top1_accuracy")),
        "top5_accuracy":                 sem.get("top5_accuracy", raw.get("top5_accuracy")),
        "macro_f1":                      sem.get("macro_f1",      raw.get("macro_f1")),
        "macro_precision":               None,
        "macro_recall":                  None,
        "avg_inference_ms_per_image":    None,
        "median_inference_ms_per_image": None,
        "rss_before_inference_mb":       None,
        "rss_after_inference_mb":        None,
        "tracemalloc_peak_mb":           None,
        "note": "Loaded from oxford_vit_evaluation_metrics.json (semantic-aligned metrics). "
                "Run without --skip-vit-eval for full timing.",
        "source_file": VIT_METRICS_PATH,
    }


def evaluate_vit_full(processor, model, raw_id2label) -> dict:
    """
    Full ViT evaluation loop. Mirrors evaluate_oxford_vit.py logic but:
      - Uses the benchmark's smaller BATCH_SIZE
      - Reports per-image inference time (forward pass only)
      - Uses build_vit_to_canonical_map for label alignment
    """
    print("\n[*] Loading Oxford Flowers 102 test split for ViT evaluation...")
    os.makedirs(DATA_DIR, exist_ok=True)
    test_ds = datasets.Flowers102(root=DATA_DIR, split="test", download=True)
    total = len(test_ds)
    canonical_classes = datasets.Flowers102.classes
    vit_to_canonical = build_vit_to_canonical_map(raw_id2label, canonical_classes)

    def vit_collate_fn(batch):
        images, labels = zip(*batch)
        inputs = processor(images=[img.convert("RGB") for img in images], return_tensors="pt")
        return inputs, torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=vit_collate_fn,
    )

    all_targets      = []
    all_mapped_preds = []
    top5_mapped_hits = 0
    inference_times_ms = []

    rss_before_mb = _get_process_rss_mb()
    tracemalloc.start()

    print(f"[*] Running ViT inference on {total:,} images "
          f"(batch={BATCH_SIZE}, device={DEVICE}) ...")

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
            targets = targets.to(DEVICE)

            t0 = time.perf_counter()
            outputs = model(**inputs)
            t1 = time.perf_counter()

            batch_ms   = (t1 - t0) * 1000.0
            per_img_ms = batch_ms / targets.size(0)
            inference_times_ms.append(per_img_ms)

            logits    = outputs.logits
            raw_preds = torch.argmax(logits, dim=-1)
            mapped    = torch.tensor(
                [vit_to_canonical.get(p.item(), p.item()) for p in raw_preds],
                device=DEVICE,
            )

            _, top5_raw = torch.topk(logits, k=5, dim=-1)
            mapped_top5 = torch.tensor(
                [[vit_to_canonical.get(idx.item(), idx.item()) for idx in row]
                 for row in top5_raw],
                device=DEVICE,
            )
            top5_mapped_hits += torch.any(
                mapped_top5 == targets.unsqueeze(1), dim=1
            ).sum().item()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_mapped_preds.extend(mapped.cpu().numpy().tolist())

            processed = (batch_idx + 1) * BATCH_SIZE
            if (batch_idx + 1) % 20 == 0 or processed >= total:
                print(f"    Processed {min(processed, total):>5}/{total} "
                      f"({min(processed, total)/total*100:>5.1f}%) "
                      f"| last batch {batch_ms:.1f}ms")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_mb = _get_process_rss_mb()

    correct  = sum(p == t for p, t in zip(all_mapped_preds, all_targets))
    top1_acc = correct / total * 100
    top5_acc = top5_mapped_hits / total * 100
    avg_ms   = sum(inference_times_ms) / len(inference_times_ms)
    med_ms   = sorted(inference_times_ms)[len(inference_times_ms) // 2]

    metrics = {
        "total_test_images":             total,
        "top1_accuracy":                 round(top1_acc, 4),
        "top5_accuracy":                 round(top5_acc, 4),
        "avg_inference_ms_per_image":    round(avg_ms, 4),
        "median_inference_ms_per_image": round(med_ms, 4),
        "rss_before_inference_mb":       round(rss_before_mb, 2),
        "rss_after_inference_mb":        round(rss_after_mb, 2),
        "tracemalloc_peak_mb":           round(peak_mem / 1024 / 1024, 2),
    }

    if _SKLEARN_OK:
        metrics["macro_precision"] = round(
            precision_score(all_targets, all_mapped_preds, average="macro", zero_division=0) * 100, 4)
        metrics["macro_recall"] = round(
            recall_score(all_targets, all_mapped_preds, average="macro", zero_division=0) * 100, 4)
        metrics["macro_f1"] = round(
            f1_score(all_targets, all_mapped_preds, average="macro", zero_division=0) * 100, 4)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Output Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(val, unit="", precision=2):
    """Format a metric value for display; return 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}{unit}"


def print_comparison_table(cnn_m: dict, vit_m: dict,
                            cnn_size_mb: float, vit_size_mb: float,
                            cnn_params: int, vit_params: int,
                            cnn_load_s: float, vit_load_s: float):
    """Print a side-by-side ASCII comparison table."""
    W = 30
    print(f"\n{DIVIDER}")
    print(f"  BENCHMARK COMPARISON SUMMARY")
    print(f"  Dataset: Oxford Flowers 102 Official Test Split")
    print(f"  Device : {DEVICE}")
    print(f"{DIVIDER}")
    print(f"{'Metric':<28} {'Custom CNN':>20} {'ViT-Base (Oxford)':>22}")
    print("-" * 72)

    rows = [
        ("Test images",          _fmt(cnn_m['total_test_images'], precision=0),
                                  _fmt(vit_m['total_test_images'], precision=0)),
        ("Parameters",           f"{cnn_params:,}",
                                  f"{vit_params:,}" if vit_params else "~86,567,642"),
        ("Checkpoint size (MB)", _fmt(cnn_size_mb),         _fmt(vit_size_mb)),
        ("Model load time (s)",  _fmt(cnn_load_s),          _fmt(vit_load_s)),
        ("",                     "",                          ""),
        ("Top-1 Accuracy (%)",   _fmt(cnn_m.get('top1_accuracy')),
                                  _fmt(vit_m.get('top1_accuracy'))),
        ("Top-5 Accuracy (%)",   _fmt(cnn_m.get('top5_accuracy')),
                                  _fmt(vit_m.get('top5_accuracy'))),
        ("Macro-Precision (%)",  _fmt(cnn_m.get('macro_precision')),
                                  _fmt(vit_m.get('macro_precision'))),
        ("Macro-Recall (%)",     _fmt(cnn_m.get('macro_recall')),
                                  _fmt(vit_m.get('macro_recall'))),
        ("Macro-F1 (%)",         _fmt(cnn_m.get('macro_f1')),
                                  _fmt(vit_m.get('macro_f1'))),
        ("",                     "",                          ""),
        ("Avg inference (ms/img)",_fmt(cnn_m.get('avg_inference_ms_per_image')),
                                   _fmt(vit_m.get('avg_inference_ms_per_image'))),
        ("Median inference (ms/img)",_fmt(cnn_m.get('median_inference_ms_per_image')),
                                      _fmt(vit_m.get('median_inference_ms_per_image'))),
    ]
    for label, cnn_val, vit_val in rows:
        if label == "":
            print()
        else:
            print(f"  {label:<26} {cnn_val:>20} {vit_val:>22}")

    print(f"\n{DIVIDER}")
    note = vit_m.get("note")
    if note:
        print(f"  [NOTE] ViT metrics source: {note[:80]}")
        print(f"         Run without --skip-vit-eval for full per-image timing.")
    print(f"{DIVIDER}\n")


def save_json(cnn_metrics: dict, vit_metrics: dict,
              cnn_size_mb: float, vit_size_mb: float,
              cnn_load_s: float, vit_load_s: float,
              cnn_params: int, vit_params: int,
              skip_vit_eval: bool):
    """Save full benchmark results to benchmark_results.json."""
    # Strip raw prediction lists from metrics before saving (large, internal)
    cnn_save = {k: v for k, v in cnn_metrics.items() if not k.startswith("_")}
    vit_save = {k: v for k, v in vit_metrics.items() if not k.startswith("_")}

    output = {
        "benchmark_info": {
            "script":        "benchmark_models.py",
            "dataset":       "Oxford Flowers 102 (official test split, 6149 images)",
            "device":        str(DEVICE),
            "batch_size":    BATCH_SIZE,
            "skip_vit_eval": skip_vit_eval,
            "timing_method": (
                "Forward pass only (model.forward()); "
                "image loading and preprocessing are excluded from per-image times. "
                "Model loading time measured separately with time.perf_counter()."
            ),
            "label_alignment": {
                "cnn": "CLASS_NAMES[argmax] matches Flowers102.classes canonical order directly.",
                "vit": (
                    "build_vit_to_canonical_map(): resolves the known 5-class cyclic rotation "
                    "at indices 16-20 and minor typos (barbeton daisy, pink-yellow dahlia?) "
                    "by matching cleaned id2label names to canonical Flowers102.classes names."
                ),
            },
        },
        "cnn": {
            "model_id":          "FlowerCNN (Custom 2-conv PyTorch CNN)",
            "checkpoint_path":   CNN_MODEL_PATH,
            "checkpoint_size_mb": cnn_size_mb,
            "parameters":        cnn_params,
            "load_time_seconds": round(cnn_load_s, 4),
            "metrics":           cnn_save,
        },
        "vit": {
            "model_id":          VIT_MODEL_ID,
            "checkpoint_size_mb": vit_size_mb,
            "parameters":        vit_params,
            "load_time_seconds": round(vit_load_s, 4) if vit_load_s else None,
            "metrics":           vit_save,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[OK] Full results saved -> {OUT_JSON}")


def save_csv(cnn_metrics: dict, vit_metrics: dict):
    """Save side-by-side comparison to benchmark_comparison.csv."""
    rows = [
        ("metric",                          "cnn_value",                       "vit_value"),
        ("top1_accuracy_pct",               cnn_metrics.get("top1_accuracy"),  vit_metrics.get("top1_accuracy")),
        ("top5_accuracy_pct",               cnn_metrics.get("top5_accuracy"),  vit_metrics.get("top5_accuracy")),
        ("macro_precision_pct",             cnn_metrics.get("macro_precision"),vit_metrics.get("macro_precision")),
        ("macro_recall_pct",                cnn_metrics.get("macro_recall"),   vit_metrics.get("macro_recall")),
        ("macro_f1_pct",                    cnn_metrics.get("macro_f1"),       vit_metrics.get("macro_f1")),
        ("avg_inference_ms_per_image",      cnn_metrics.get("avg_inference_ms_per_image"),
                                             vit_metrics.get("avg_inference_ms_per_image")),
        ("median_inference_ms_per_image",   cnn_metrics.get("median_inference_ms_per_image"),
                                             vit_metrics.get("median_inference_ms_per_image")),
        ("rss_before_inference_mb",         cnn_metrics.get("rss_before_inference_mb"),
                                             vit_metrics.get("rss_before_inference_mb")),
        ("rss_after_inference_mb",          cnn_metrics.get("rss_after_inference_mb"),
                                             vit_metrics.get("rss_after_inference_mb")),
        ("tracemalloc_peak_mb",             cnn_metrics.get("tracemalloc_peak_mb"),
                                             vit_metrics.get("tracemalloc_peak_mb")),
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"[OK] Comparison table saved -> {OUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Custom CNN vs ViT on Oxford Flowers 102 test split."
    )
    parser.add_argument(
        "--skip-vit-eval",
        action="store_true",
        default=False,
        help=(
            "Load ViT metrics from oxford_vit_evaluation_metrics.json instead of "
            "re-running the full ViT evaluation (~22 min on CPU). "
            "Per-image inference timing will be N/A in this mode."
        ),
    )
    args = parser.parse_args()

    print(f"\n{DIVIDER}")
    print("  FLOWER CLASSIFICATION BENCHMARK: CNN vs ViT")
    print(f"  Dataset : Oxford Flowers 102 (official test split)")
    print(f"  Device  : {DEVICE}")
    print(f"  Mode    : {'Skip ViT eval (load from JSON)' if args.skip_vit_eval else 'Full evaluation (both models)'}")
    print(f"{DIVIDER}\n")

    # ── CNN: Load ──────────────────────────────────────────────────────────────
    print("[1/4] Loading Custom CNN ...")
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"[ERROR] CNN checkpoint not found: {CNN_MODEL_PATH}")
        print("        Make sure model/flower_classification_model.pth exists.")
        sys.exit(1)

    cnn_model, cnn_load_s, cnn_size_mb = load_cnn()
    cnn_params = sum(p.numel() for p in cnn_model.parameters())

    # ── ViT: Load (or skip) ────────────────────────────────────────────────────
    vit_processor = vit_model = vit_raw_id2label = None
    vit_load_s  = None
    vit_params  = None
    vit_size_mb = -1.0

    if not args.skip_vit_eval:
        print("\n[2/4] Loading Vision Transformer ...")
        if not _TRANSFORMERS_OK:
            print("[ERROR] 'transformers' library is required for ViT evaluation.")
            print("        Install: pip install transformers")
            print("        Or run with --skip-vit-eval to use cached metrics.")
            sys.exit(1)
        try:
            vit_processor, vit_model, vit_raw_id2label, vit_load_s, vit_size_mb = load_vit()
            vit_params = sum(p.numel() for p in vit_model.parameters())
        except Exception as e:
            print(f"[ERROR] Failed to load ViT: {e}")
            sys.exit(1)
    else:
        if not os.path.exists(VIT_METRICS_PATH):
            print(f"[ERROR] --skip-vit-eval requested but metrics file not found:")
            print(f"        {VIT_METRICS_PATH}")
            print("        Run evaluate_oxford_vit.py first, or drop --skip-vit-eval.")
            sys.exit(1)
        print(f"\n[2/4] Skipping ViT evaluation — loading from: {VIT_METRICS_PATH}")
        vit_params  = 86_567_642   # Standard ViT-Base parameter count
        vit_size_mb = 327.6        # From HF cache inspection

    # ── CNN Evaluation ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Evaluating Custom CNN on Oxford Flowers 102 test split ...")
    bench_start = time.perf_counter()
    cnn_metrics = evaluate_cnn(cnn_model)
    cnn_eval_s  = time.perf_counter() - bench_start

    print(f"\n  CNN evaluation complete in {cnn_eval_s:.1f}s")
    print(f"  Top-1 Accuracy : {_fmt(cnn_metrics.get('top1_accuracy'))}%")
    print(f"  Top-5 Accuracy : {_fmt(cnn_metrics.get('top5_accuracy'))}%")
    print(f"  Macro-F1       : {_fmt(cnn_metrics.get('macro_f1'))}%")
    print(f"  Avg inference  : {_fmt(cnn_metrics.get('avg_inference_ms_per_image'))} ms/image")

    # ── ViT Evaluation ─────────────────────────────────────────────────────────
    if args.skip_vit_eval:
        print(f"\n[4/4] Loading ViT metrics from file ...")
        vit_metrics = evaluate_vit_from_file()
    else:
        print(f"\n[4/4] Evaluating Vision Transformer on Oxford Flowers 102 test split ...")
        vit_bench_start = time.perf_counter()
        vit_metrics = evaluate_vit_full(vit_processor, vit_model, vit_raw_id2label)
        vit_eval_s  = time.perf_counter() - vit_bench_start
        print(f"\n  ViT evaluation complete in {vit_eval_s:.1f}s")
        print(f"  Top-1 Accuracy : {_fmt(vit_metrics.get('top1_accuracy'))}%")
        print(f"  Top-5 Accuracy : {_fmt(vit_metrics.get('top5_accuracy'))}%")
        print(f"  Macro-F1       : {_fmt(vit_metrics.get('macro_f1'))}%")
        print(f"  Avg inference  : {_fmt(vit_metrics.get('avg_inference_ms_per_image'))} ms/image")

    # ── Print Comparison Table ─────────────────────────────────────────────────
    print_comparison_table(
        cnn_metrics, vit_metrics,
        cnn_size_mb, vit_size_mb,
        cnn_params,  vit_params if vit_params else 0,
        cnn_load_s,  vit_load_s if vit_load_s else 0.0,
    )

    # ── Save Outputs ───────────────────────────────────────────────────────────
    save_json(
        cnn_metrics, vit_metrics,
        cnn_size_mb, vit_size_mb,
        cnn_load_s,  vit_load_s if vit_load_s else 0.0,
        cnn_params,  vit_params if vit_params else 0,
        args.skip_vit_eval,
    )
    save_csv(cnn_metrics, vit_metrics)

    print("\n[DONE] Benchmark complete.")
    print(f"  Results: {OUT_JSON}")
    print(f"  Table  : {OUT_CSV}")
    print()
    print("  IMPORTANT: These metrics reflect performance on the Oxford Flowers 102")
    print("  official test set only. Real-world accuracy may differ based on image")
    print("  quality, lighting, and subject framing.")


if __name__ == "__main__":
    main()

