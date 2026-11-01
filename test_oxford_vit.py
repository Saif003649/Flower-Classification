"""
Flower Classification — Oxford 102 ViT Verification Script
Model: loretyan/vit-base-oxford-flowers-102

Verifies:
1. Exact class count (102 classes)
2. Alignment between model.config.id2label and existing CLASS_NAMES in app.py
3. Inference on sample images in static/uploads/ using torch.inference_mode()
4. Displays top-3 predictions and confidence percentages
"""

import os
import sys

# 1. Dependency checks
try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ModuleNotFoundError as e:
    print(f"\n[✗] Missing required library: {e.name}")
    print("Please install transformers before running this script:")
    print("    pip install transformers")
    sys.exit(1)

# 2. Existing 102 Oxford Flower Class Names from app.py
PROJECT_CLASS_NAMES = [
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

MODEL_ID = "loretyan/vit-base-oxford-flowers-102"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIVIDER = "=" * 70


def main():
    print(f"\n{DIVIDER}")
    print("  OXFORD 102 ViT VERIFICATION: loretyan/vit-base-oxford-flowers-102")
    print(f"{DIVIDER}")
    print(f"Device        : {DEVICE}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Expected Size : 102 classes")

    # 3. Load Model and Processor
    print(f"\n{DIVIDER}")
    print("  STEP 1: LOADING MODEL AND PROCESSOR")
    print(DIVIDER)

    try:
        print(f"[*] Downloading / loading image processor: {MODEL_ID} ...")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        print("    [✓] AutoImageProcessor loaded successfully.")

        print(f"[*] Downloading / loading classification model: {MODEL_ID} ...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        model.to(DEVICE)
        model.eval()
        print("    [✓] AutoModelForImageClassification loaded and set to eval mode.")

    except Exception as e:
        print(f"\n[✗] Failed to load model or processor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify Class Count and Label Mapping
    print(f"\n{DIVIDER}")
    print("  STEP 2: LABEL VERIFICATION AND ALIGNMENT CHECK")
    print(DIVIDER)

    num_labels = getattr(model.config, "num_labels", None)
    id2label = getattr(model.config, "id2label", {})

    print(f"Model num_labels: {num_labels}")
    print(f"id2label count  : {len(id2label)}")

    if num_labels != 102:
        print(f"[✗] ERROR: Expected 102 classes, but model has {num_labels} classes.")
        sys.exit(1)

    # Check whether id2label is indexed by integer or string representation of integer
    # Normalize to int keys
    norm_id2label = {}
    for k, v in id2label.items():
        try:
            norm_id2label[int(k)] = str(v)
        except ValueError:
            norm_id2label[k] = str(v)

    # Compare against project CLASS_NAMES
    matches = 0
    case_differences = 0
    mismatches = []

    for idx in range(102):
        expected_name = PROJECT_CLASS_NAMES[idx]
        actual_name = norm_id2label.get(idx, None)

        if actual_name is None:
            mismatches.append((idx, expected_name, "<MISSING>"))
            continue

        if actual_name == expected_name:
            matches += 1
        elif actual_name.lower().strip() == expected_name.lower().strip():
            case_differences += 1
        else:
            mismatches.append((idx, expected_name, actual_name))

    print(f"\nComparison with project's CLASS_NAMES (102 classes):")
    print(f"  - Exact matches        : {matches} / 102")
    print(f"  - Case/format matches  : {case_differences} / 102")
    print(f"  - Complete mismatches  : {len(mismatches)} / 102")

    if mismatches:
        print("\nMismatched indices detected:")
        for idx, exp, act in mismatches[:10]:
            print(f"    Index {idx:<3}: Expected='{exp}', Actual='{act}'")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more.")
    else:
        print("\n[✓] PERFECT ALIGNMENT: Model label order matches project CLASS_NAMES exactly.")

    # Print sample labels
    print("\nSample mapped labels from model (first 5 and last 5):")
    for i in list(range(5)) + list(range(97, 102)):
        print(f"    Index {i:<3} -> '{norm_id2label.get(i)}'")

    # 5. Inference on Uploaded Images
    print(f"\n{DIVIDER}")
    print("  STEP 3: INFERENCE ON TEST IMAGES (static/uploads/)")
    print(DIVIDER)

    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = []
    if os.path.exists(UPLOADS_DIR):
        image_files = [
            f for f in sorted(os.listdir(UPLOADS_DIR))
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

    if not image_files:
        print("[!] No images found in static/uploads/. Using synthetic test image.")
        image_files = ["<synthetic>"]

    processed_count = 0
    for fname in image_files:
        if fname == "<synthetic>":
            img = Image.new("RGB", (224, 224), color=(200, 150, 100))
            display_name = "Synthetic RGB image (224x224)"
        else:
            img_path = os.path.join(UPLOADS_DIR, fname)
            try:
                with Image.open(img_path) as test_open:
                    test_open.verify()
                img = Image.open(img_path).convert("RGB")
                display_name = fname
            except Exception as e:
                print(f"[-] Skipping unreadable file {fname}: {e}")
                continue

        processed_count += 1
        print(f"\nImage [{processed_count}]: {display_name}")

        try:
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)

            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]

            top_probs, top_indices = torch.topk(probs, k=3)

            print(f"  {'Rank':<6} {'Index':<8} {'Predicted Flower':<28} {'Confidence':>12}")
            print(f"  {'-'*6} {'-'*8} {'-'*28} {'-'*12}")

            for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
                c_idx = idx.item()
                lbl = norm_id2label.get(c_idx, f"Class {c_idx}")
                print(f"  {rank:<6} {c_idx:<8} {lbl.title():<28} {p.item()*100:>11.2f}%")

        except Exception as e:
            print(f"  [✗] Inference error: {e}")

    # 6. Evaluation Protocol Guidelines
    print(f"\n{DIVIDER}")
    print("  STEP 4: RIGOROUS EVALUATION GUIDELINES (ACCURACY & MACRO-F1)")
    print(DIVIDER)
    print("To benchmark this model properly on the full test set without data leakage:")
    print("1. Dataset source: Use torchvision.datasets.Flowers102(root='./data', split='test')")
    print("   or Hugging Face datasets: load_dataset('oxford_flowers102', split='test')")
    print("2. Do NOT train, tune hyperparameters, or select thresholds on the test split.")
    print("3. Metrics to compute across all test samples:")
    print("   - Top-1 Accuracy: (correct_predictions / total_samples) * 100")
    print("   - Top-5 Accuracy: fraction where ground truth is within top-5 logits")
    print("   - Macro-F1: sklearn.metrics.f1_score(y_true, y_pred, average='macro')")
    print("     (Computes unweighted mean F1 across all 102 classes to penalize class imbalance)")
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    main()

