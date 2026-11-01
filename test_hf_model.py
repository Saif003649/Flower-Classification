"""
Test script for Hugging Face Vision Transformer model:
Akshay0706/Flower-Image-Classification-Model

Tests:
1. Loading AutoImageProcessor and AutoModelForImageClassification
2. Verifying model.config.id2label and model.config.label2id mappings
3. Running inference with torch.inference_mode() on a sample image or synthetic fallback
"""

import os
import sys

# 1. Dependency check
try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ModuleNotFoundError as e:
    print(f"\n[!] Missing dependency: {e.name}")
    print("Please install the required libraries before running this script:")
    print("    pip install transformers")
    sys.exit(1)

# 2. Configuration
MODEL_ID = "Akshay0706/Flower-Image-Classification-Model"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIVIDER = "=" * 65

def main():
    print(f"\n{DIVIDER}")
    print("  HUGGING FACE MODEL INTEGRATION TEST (PHASE 1)")
    print(f"{DIVIDER}")
    print(f"Target Model Repository : {MODEL_ID}")
    print(f"PyTorch Device           : {DEVICE}")
    print(f"PyTorch Version          : {torch.__version__}")

    # 3. Load Model and Processor
    print(f"\n{DIVIDER}")
    print("  LOADING PROCESSOR AND MODEL")
    print(DIVIDER)

    try:
        print("[*] Fetching/Loading AutoImageProcessor...")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        print("    [✓] AutoImageProcessor loaded successfully.")

        print("[*] Fetching/Loading AutoModelForImageClassification...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        model.to(DEVICE)
        model.eval()
        print("    [✓] AutoModelForImageClassification loaded and set to eval mode.")

    except Exception as e:
        print(f"\n[✗] Failed to load model or processor from Hugging Face: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Inspect Model Architecture and Label Mappings
    print(f"\n{DIVIDER}")
    print("  MODEL CONFIGURATION & LABEL INSPECTION")
    print(DIVIDER)
    
    num_labels = getattr(model.config, "num_labels", None)
    id2label = getattr(model.config, "id2label", None)
    label2id = getattr(model.config, "label2id", None)

    print(f"Architecture    : {model.__class__.__name__}")
    print(f"Number of labels: {num_labels}")
    print(f"id2label mapping: {id2label}")
    print(f"label2id mapping: {label2id}")

    # Verify if labels are descriptive flower names or generic tokens
    is_generic_mapping = False
    if isinstance(id2label, dict):
        sample_val = str(list(id2label.values())[0])
        if sample_val.upper().startswith("LABEL_") or sample_val.isdigit():
            is_generic_mapping = True

    if is_generic_mapping:
        print("\n[!] WARNING: `id2label` contains generic or raw integer labels rather than flower names.")
        print("    An explicit class mapping dictionary will be required before app integration.")
    else:
        print("\n[✓] `id2label` appears to contain mapped class names.")

    # 5. Image Selection (from static/uploads/ or synthetic fallback)
    print(f"\n{DIVIDER}")
    print("  SELECTING TEST IMAGE")
    print(DIVIDER)

    valid_exts = {".jpg", ".jpeg", ".png"}
    test_image_path = None

    if os.path.exists(UPLOADS_DIR):
        candidates = [
            os.path.join(UPLOADS_DIR, f)
            for f in os.listdir(UPLOADS_DIR)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        if candidates:
            # Pick the first readable image
            for candidate in candidates:
                try:
                    with Image.open(candidate) as img:
                        img.verify()
                    test_image_path = candidate
                    break
                except Exception:
                    continue

    if test_image_path:
        print(f"Loaded test image: {os.path.basename(test_image_path)}")
        test_img = Image.open(test_image_path).convert("RGB")
    else:
        print("No valid test images found in static/uploads/. Creating a synthetic RGB test image...")
        test_img = Image.new("RGB", (224, 224), color=(220, 180, 100))
        test_image_path = "<synthetic_test_image>"

    # 6. Inference Execution
    print(f"\n{DIVIDER}")
    print("  RUNNING INFERENCE")
    print(DIVIDER)

    try:
        inputs = processor(images=test_img, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        top_prob, top_idx = torch.topk(probabilities, k=min(5, len(probabilities)))

        print("\nPrediction Results:")
        print(f"  {'Rank':<6} {'Index':<8} {'Label Name':<25} {'Confidence':>12}")
        print(f"  {'-'*6} {'-'*8} {'-'*25} {'-'*12}")

        for rank, (p, idx) in enumerate(zip(top_prob, top_idx), start=1):
            class_id = idx.item()
            label_name = id2label.get(class_id, f"Class {class_id}") if isinstance(id2label, dict) else f"Class {class_id}"
            conf_percent = p.item() * 100
            print(f"  {rank:<6} {class_id:<8} {str(label_name):<25} {conf_percent:>11.2f}%")

        top_class_id = top_idx[0].item()
        top_label = id2label.get(top_class_id, str(top_class_id)) if isinstance(id2label, dict) else str(top_class_id)
        print(f"\n[✓] Top Prediction : {top_label} ({top_prob[0].item() * 100:.2f}%)")
        print("[✓] Inference completed successfully.")

    except Exception as e:
        print(f"\n[✗] Inference error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

