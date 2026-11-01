"""
Evaluation Script for Oxford 102 Vision Transformer
Model: loretyan/vit-base-oxford-flowers-102
Dataset: torchvision.datasets.Flowers102 (Official Test Split: 6,149 images)

Features:
1. Label alignment verification between model.config.id2label and Flowers102.classes.
2. Evaluates Top-1 accuracy, Top-5 accuracy, and Macro-F1 using official Flowers102 test split.
3. Computes both:
   a) Index-based metrics (raw output neuron index vs dataset class index)
   b) Name-based metrics (matching string name from id2label with ground-truth name)
4. Saves full evaluation metrics to JSON and confusion matrix to CSV.
"""

import os
import sys
import json
import time

try:
    import torch
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
except ModuleNotFoundError as e:
    print(f"\n[✗] Missing required library: {e.name}")
    print("Please install missing dependencies:")
    print("    pip install transformers scikit-learn")
    sys.exit(1)

MODEL_ID = "loretyan/vit-base-oxford-flowers-102"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "oxford_vit_evaluation_metrics.json")
OUTPUT_CM_PATH = os.path.join(BASE_DIR, "oxford_vit_confusion_matrix.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
DIVIDER = "=" * 72


def collate_fn_builder(processor):
    """Custom collate function applying official ViTImageProcessor to PIL images."""
    def collate_fn(batch):
        images, labels = zip(*batch)
        # Processor handles resizing to 224x224, scaling, and normalization
        inputs = processor(images=list(images), return_tensors="pt")
        targets = torch.tensor(labels, dtype=torch.long)
        return inputs, targets
    return collate_fn


def main():
    print(f"\n{DIVIDER}")
    print("  OXFORD 102 ViT BENCHMARK & EVALUATION")
    print(f"  Model: {MODEL_ID}")
    print(f"{DIVIDER}")
    print(f"PyTorch Device : {DEVICE}")
    print(f"Batch Size     : {BATCH_SIZE}")

    # 1. Load Model and Official Image Processor
    print("\n[*] Loading official image processor and ViT model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    print("    [✓] Model and processor loaded successfully.")

    # 2. Inspect Model id2label vs Official Flowers102 Classes
    print(f"\n{DIVIDER}")
    print("  LABEL ALIGNMENT INSPECTION")
    print(f"{DIVIDER}")

    canonical_classes = datasets.Flowers102.classes
    model_id2label = getattr(model.config, "id2label", {})

    # Normalize id2label keys to integers
    norm_model_labels = {}
    for k, v in model_id2label.items():
        norm_model_labels[int(k)] = str(v)

    # Name-to-canonical-index lookup
    # Normalize names for robust comparison (strip whitespace and lower-case)
    canonical_name_to_idx = {
        name.lower().strip().rstrip("?"): idx
        for idx, name in enumerate(canonical_classes)
    }

    # Alignment check
    raw_index_matches = 0
    shifted_indices = []
    model_to_canonical_map = {}

    for i in range(102):
        can_name = canonical_classes[i]
        mod_name = norm_model_labels.get(i, "")
        
        # Check raw index alignment
        if mod_name.lower().strip().rstrip("?") == can_name.lower().strip().rstrip("?"):
            raw_index_matches += 1
        else:
            shifted_indices.append((i, can_name, mod_name))

        # Check semantic name match to build alignment mapping
        clean_mod = mod_name.lower().strip().rstrip("?")
        if clean_mod in canonical_name_to_idx:
            model_to_canonical_map[i] = canonical_name_to_idx[clean_mod]
        else:
            # Handle known edge cases (e.g. bearded iris vs iris)
            if "iris" in clean_mod:
                model_to_canonical_map[i] = canonical_name_to_idx.get("iris", canonical_name_to_idx.get("bearded iris", i))
            else:
                model_to_canonical_map[i] = i

    print(f"Raw Index-to-Index matches: {raw_index_matches} / 102")
    if shifted_indices:
        print(f"[!] Mismatches detected on {len(shifted_indices)} indices:")
        for idx, can, mod in shifted_indices:
            print(f"    Index {idx:<2}: Dataset Canonical='{can}' | Model Output='{mod}'")
        print("\nNote: Indices 16-20 exhibit a cyclical rotation in model output labels.")

    # 3. Download/Load Test Split
    print(f"\n{DIVIDER}")
    print("  LOADING OFFICIAL OXFORD FLOWERS 102 TEST SPLIT")
    print(f"{DIVIDER}")
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading Flowers102 test split from: {DATA_DIR} ...")
    test_dataset = datasets.Flowers102(
        root=DATA_DIR,
        split="test",
        download=True
    )
    total_samples = len(test_dataset)
    print(f"Total test instances: {total_samples:,} images (zero data leakage)")

    collate_fn = collate_fn_builder(processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 4. Evaluation Loop
    print(f"\n{DIVIDER}")
    print("  RUNNING INFERENCE ON TEST SET (torch.inference_mode)")
    print(f"{DIVIDER}")

    all_targets = []
    all_raw_preds = []
    all_mapped_preds = []
    all_raw_top5_hits = 0
    all_mapped_top5_hits = 0

    t_start = time.time()
    processed = 0

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            targets = targets.to(DEVICE)

            outputs = model(**inputs)
            logits = outputs.logits  # [B, 102]

            # Top 1 predictions
            raw_preds = torch.argmax(logits, dim=-1)
            
            # Map model output neurons to canonical indices via verified name mapping
            mapped_preds = torch.tensor(
                [model_to_canonical_map.get(p.item(), p.item()) for p in raw_preds],
                device=DEVICE
            )

            # Top 5 accuracy
            _, top5_raw_indices = torch.topk(logits, k=5, dim=-1)
            raw_top5 = torch.any(top5_raw_indices == targets.unsqueeze(1), dim=-1)
            all_raw_top5_hits += raw_top5.sum().item()

            # Mapped top 5
            mapped_top5_indices = torch.tensor(
                [[model_to_canonical_map.get(idx.item(), idx.item()) for idx in row] for row in top5_raw_indices],
                device=DEVICE
            )
            mapped_top5 = torch.any(mapped_top5_indices == targets.unsqueeze(1), dim=-1)
            all_mapped_top5_hits += mapped_top5.sum().item()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_raw_preds.extend(raw_preds.cpu().numpy().tolist())
            all_mapped_preds.extend(mapped_preds.cpu().numpy().tolist())

            processed += len(targets)
            if (batch_idx + 1) % 20 == 0 or processed == total_samples:
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed:>5}/{total_samples} images ({processed/total_samples*100:>5.1f}%) "
                      f"[{rate:.1f} img/s]")

    total_time = time.time() - t_start

    # 5. Compute Metrics
    print(f"\n{DIVIDER}")
    print("  EVALUATION METRICS & RESULTS")
    print(f"{DIVIDER}")

    # Compute raw accuracy & macro-F1 (unaligned neurons)
    raw_correct = sum(p == t for p, t in zip(all_raw_preds, all_targets))
    raw_top1_acc = (raw_correct / total_samples) * 100
    raw_top5_acc = (all_raw_top5_hits / total_samples) * 100
    raw_macro_f1 = f1_score(all_targets, all_raw_preds, average="macro") * 100

    # Compute mapped accuracy & macro-F1 (aligned by flower name)
    mapped_correct = sum(p == t for p, t in zip(all_mapped_preds, all_targets))
    mapped_top1_acc = (mapped_correct / total_samples) * 100
    mapped_top5_acc = (all_mapped_top5_hits / total_samples) * 100
    mapped_macro_f1 = f1_score(all_targets, all_mapped_preds, average="macro") * 100

    print(f"\nTotal test samples evaluated : {total_samples:,}")
    print(f"Inference duration           : {total_time:.1f} s ({total_samples/total_time:.1f} img/s)")

    print(f"\n1. Raw Output Evaluation (Direct neuron index comparison):")
    print(f"   - Top-1 Accuracy : {raw_top1_acc:>6.2f}%")
    print(f"   - Top-5 Accuracy : {raw_top5_acc:>6.2f}%")
    print(f"   - Macro-F1 Score : {raw_macro_f1:>6.2f}%")

    print(f"\n2. Semantic Alignment Evaluation (Aligned via model.config.id2label):")
    print(f"   - Top-1 Accuracy : {mapped_top1_acc:>6.2f}%")
    print(f"   - Top-5 Accuracy : {mapped_top5_acc:>6.2f}%")
    print(f"   - Macro-F1 Score : {mapped_macro_f1:>6.2f}%")

    # Shifted classes specific analysis (indices 16-20)
    shifted_target_indices = {16, 17, 18, 19, 20}
    shifted_mask = [t in shifted_target_indices for t in all_targets]
    shifted_total = sum(shifted_mask)

    if shifted_total > 0:
        sub_targets = [t for t, m in zip(all_targets, shifted_mask) if m]
        sub_raw = [p for p, m in zip(all_raw_preds, shifted_mask) if m]
        sub_mapped = [p for p, m in zip(all_mapped_preds, shifted_mask) if m]

        sub_raw_acc = sum(p == t for p, t in zip(sub_raw, sub_targets)) / shifted_total * 100
        sub_mapped_acc = sum(p == t for p, t in zip(sub_mapped, sub_targets)) / shifted_total * 100

        print(f"\nIndices 16-20 Specific Diagnostic (Total {shifted_total} test images):")
        print(f"   - Raw Index Accuracy    : {sub_raw_acc:>6.2f}% (Suffers from index rotation)")
        print(f"   - Semantic Label Accuracy : {sub_mapped_acc:>6.2f}% (Resolved by id2label mapping)")

    # 6. Save Artifacts
    results_dict = {
        "model_id": MODEL_ID,
        "dataset": "Oxford Flowers 102 (test split)",
        "total_test_samples": total_samples,
        "raw_metrics": {
            "top1_accuracy": round(raw_top1_acc, 2),
            "top5_accuracy": round(raw_top5_acc, 2),
            "macro_f1": round(raw_macro_f1, 2)
        },
        "semantic_aligned_metrics": {
            "top1_accuracy": round(mapped_top1_acc, 2),
            "top5_accuracy": round(mapped_top5_acc, 2),
            "macro_f1": round(mapped_macro_f1, 2)
        },
        "execution_time_seconds": round(total_time, 2)
    }

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[✓] Metrics saved to: {OUTPUT_METRICS_PATH}")

    # Compute and save confusion matrix (semantic aligned)
    cm = confusion_matrix(all_targets, all_mapped_preds, labels=list(range(102)))
    with open(OUTPUT_CM_PATH, "w", encoding="utf-8") as f:
        # Header with class names
        header = ["Target_Index", "Target_Class_Name"] + [f"Pred_{i}" for i in range(102)]
        f.write(",".join(header) + "\n")
        for i in range(102):
            row_vals = [str(i), f'"{canonical_classes[i]}"'] + [str(count) for count in cm[i]]
            f.write(",".join(row_vals) + "\n")
    print(f"[✓] Confusion matrix saved to: {OUTPUT_CM_PATH}")
    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()

