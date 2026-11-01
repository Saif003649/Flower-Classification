"""
Flower Classification Web Application
Flask Backend Server

Supported Models:
1. Vision Transformer (ViT-Base) — loretyan/vit-base-oxford-flowers-102
2. Custom CNN (PyTorch) — model/flower_classification_model.pth

Dataset: 102 Oxford Flower Categories
"""

import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ─────────────────────────────────────────────────────────────
#  App Configuration
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "flower_secret_key_2024"

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "flower_classification_model.pth")
VIT_MODEL_ID   = "loretyan/vit-base-oxford-flowers-102"

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  Device Configuration
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
#  Class Labels  ← 102 Oxford Flower categories (Canonical)
# ─────────────────────────────────────────────────────────────
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
NUM_CLASSES = len(CLASS_NAMES)   # 102

# ─────────────────────────────────────────────────────────────
#  Model 1: Custom CNN Architecture (Preserved Unchanged)
# ─────────────────────────────────────────────────────────────
class FlowerCNN(nn.Module):
    def __init__(self, num_classes: int = 102):
        super(FlowerCNN, self).__init__()

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
        x = self.classifier(x)
        return x

# CNN Image Preprocessing (224×224, ImageNet normalization)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_cnn_model():
    """Load the pretrained Custom CNN model once at startup."""
    try:
        m = FlowerCNN(num_classes=NUM_CLASSES)
        state = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=True)

        if isinstance(state, dict) and "model_state_dict" in state:
            m.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            m.load_state_dict(state["state_dict"])
        else:
            m.load_state_dict(state)

        m.to(device)
        m.eval()
        print(f"[OK] CNN Model loaded successfully -> {CNN_MODEL_PATH}")
        return m
    except FileNotFoundError:
        print(f"[WARN] CNN model file not found at: {CNN_MODEL_PATH}")
        return None
    except Exception as e:
        print(f"[WARN] Error loading CNN model: {e}")
        return None

# ─────────────────────────────────────────────────────────────
#  Model 2: Vision Transformer (ViT-Base) Loader
# ─────────────────────────────────────────────────────────────
def clean_label_name(raw_name: str) -> str:
    """Clean label names by fixing known typos and removing artifacts."""
    name = str(raw_name).strip().rstrip("?")
    if name.lower() == "barbeton daisy":
        return "barberton daisy"
    return name

def load_vit_model():
    """Load the pretrained Hugging Face ViT model and processor once at startup."""
    try:
        print(f"[*] Loading Vision Transformer: {VIT_MODEL_ID} ...")
        processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID)
        model.to(device)
        model.eval()

        # Build clean id2label lookup
        id2label = {}
        for k, v in model.config.id2label.items():
            id2label[int(k)] = clean_label_name(v)

        print(f"[OK] ViT Model loaded successfully ({len(id2label)} classes).")
        return processor, model, id2label
    except Exception as e:
        print(f"[WARN] Failed to load ViT model ({VIT_MODEL_ID}): {e}")
        return None, None, {}

# ─────────────────────────────────────────────────────────────
#  Load Both Models at Startup (Loaded ONCE, not per request)
# ─────────────────────────────────────────────────────────────
cnn_model = load_cnn_model()
vit_processor, vit_model, vit_id2label = load_vit_model()

# ─────────────────────────────────────────────────────────────
#  Inference Pipelines
# ─────────────────────────────────────────────────────────────
def predict_cnn(image_path: str):
    """Run inference using the Custom CNN model. Returns top-3 predictions."""
    if cnn_model is None:
        raise RuntimeError("Custom CNN model is not available.")

    img = Image.open(image_path).convert("RGB")
    tensor = cnn_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(3, len(probs)))
    top3 = []
    for p, idx in zip(top_probs, top_indices):
        class_name = CLASS_NAMES[idx.item()]
        conf_pct = round(p.item() * 100, 2)
        top3.append({"class_name": class_name, "confidence": conf_pct})

    return top3

def predict_vit(image_path: str):
    """Run inference using the Vision Transformer model. Returns top-3 predictions."""
    if vit_model is None or vit_processor is None:
        raise RuntimeError("Vision Transformer model is not available.")

    img = Image.open(image_path).convert("RGB")
    inputs = vit_processor(images=img, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = vit_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(3, len(probs)))
    top3 = []
    for p, idx in zip(top_probs, top_indices):
        class_id = idx.item()
        # Decode using the verified ViT id2label mapping (NOT raw CNN index)
        class_name = vit_id2label.get(class_id, f"Class {class_id}")
        conf_pct = round(p.item() * 100, 2)
        top3.append({"class_name": class_name, "confidence": conf_pct})

    return top3

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def cleanup_old_uploads(max_age_seconds: int = 3600):
    """
    Remove temporary user uploads (prefixed with 'user_upload_' or 'temp_') older than
    max_age_seconds to ensure user-uploaded images are not stored permanently on the server.
    Preserves any static project assets.
    """
    try:
        now = time.time()
        for fname in os.listdir(UPLOAD_FOLDER):
            if not (fname.startswith("user_upload_") or fname.startswith("temp_")):
                continue
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath) > max_age_seconds):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Home / Upload page with model selector."""
    vit_available = vit_model is not None
    cnn_available = cnn_model is not None
    return render_template(
        "index.html",
        vit_available=vit_available,
        cnn_available=cnn_available,
    )

@app.route("/predict", methods=["POST"])
def predict_route():
    """Receive uploaded image and model choice -> predict -> render result."""
    # 1. File present?
    if "file" not in request.files:
        flash("No file part in the request. Please select an image.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No image selected. Please upload a flower image.", "error")
        return redirect(url_for("index"))

    # 2. Valid extension?
    if not allowed_file(file.filename):
        flash("Unsupported file format. Please upload a JPG, JPEG, or PNG image.", "error")
        return redirect(url_for("index"))

    # 3. Model selection
    model_choice = request.form.get("model_choice", "vit").lower().strip()
    if model_choice not in {"vit", "cnn"}:
        model_choice = "vit"

    try:
        # Run periodic cleanup of old temporary user uploads
        cleanup_old_uploads()

        # 4. Save securely as temporary user session upload
        filename = f"user_upload_{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # 5. Image integrity check
        try:
            with Image.open(save_path) as img:
                img.verify()
        except Exception:
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass
            flash("The uploaded file is corrupted or not a valid image.", "error")
            return redirect(url_for("index"))

        # 6. Execute selected prediction pipeline
        if model_choice == "vit":
            if vit_model is None:
                flash("Vision Transformer model is currently unavailable. Using Custom CNN instead.", "warning")
                top3 = predict_cnn(save_path)
                model_name = "Custom CNN (PyTorch)"
            else:
                top3 = predict_vit(save_path)
                model_name = "Vision Transformer (ViT-Base)"
        else:
            if cnn_model is None:
                flash("Custom CNN model is currently unavailable. Using Vision Transformer instead.", "warning")
                top3 = predict_vit(save_path)
                model_name = "Vision Transformer (ViT-Base)"
            else:
                top3 = predict_cnn(save_path)
                model_name = "Custom CNN (PyTorch)"

        image_url = url_for("static", filename=f"uploads/{filename}")
        primary_class = top3[0]["class_name"].title()
        primary_conf = top3[0]["confidence"]

        return render_template(
            "result.html",
            image_url=image_url,
            class_name=primary_class,
            confidence=primary_conf,
            top3=top3,
            model_name=model_name,
            model_choice=model_choice,
        )

    except Exception as e:
        # Server console log only; never expose file paths or stack traces in flash
        print(f"[ERROR] Prediction failed: {e}")
        flash("An unexpected error occurred while analyzing the image. Please verify it is a valid flower photo and try again.", "error")
        return redirect(url_for("index"))

@app.errorhandler(413)
def request_entity_too_large(error):
    """Graceful handler for uploads exceeding MAX_CONTENT_LENGTH."""
    flash("The uploaded image exceeds the 16 MB limit. Please upload a smaller image file.", "error")
    return redirect(url_for("index")), 303

@app.route("/about")
def about():
    """About page with technical architecture details."""
    return render_template(
        "about.html",
        vit_available=vit_model is not None,
        cnn_available=cnn_model is not None,
    )

# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
