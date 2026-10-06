"""
Flower Classification Web Application
Flask Backend Server

Model: Custom CNN  (2 conv blocks + 2 FC layers)
Input: 224 x 224 RGB
Output: 102 classes
File: flower_classification_model.pth
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────────
#  App Configuration
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "flower_secret_key_2024"

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "flower_classification_model.pth")

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  Class Labels  ← 102 Oxford Flower categories (in training order)
#  Change ONLY if your model was trained on a different label set.
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
#  Model Architecture  ← Must exactly match how it was trained
#
#  Detected from .pth weights:
#   features.0  Conv2d(3,  16, kernel=3) + ReLU + MaxPool2d(2)
#   features.3  Conv2d(16, 32, kernel=3) + ReLU + MaxPool2d(2)
#   classifier.0  Linear(100352, 128)    100352 = 32 * 56 * 56
#   classifier.2  Linear(128, 102)
# ─────────────────────────────────────────────────────────────
class FlowerCNN(nn.Module):
    def __init__(self, num_classes: int = 102):
        super(FlowerCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # → [B, 16, 224, 224]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → [B, 16, 112, 112]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → [B, 32, 112, 112]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → [B, 32,  56,  56]
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),                 # 100352 → 128
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),                   # 128 → 102
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
#  Load Model Once at Startup
# ─────────────────────────────────────────────────────────────
def load_model():
    """Load the pretrained CNN model from disk. Called once at startup."""
    try:
        m = FlowerCNN(num_classes=NUM_CLASSES)
        state = torch.load(MODEL_PATH, map_location=device)

        # Handle raw state_dict OR wrapped checkpoint formats
        if isinstance(state, dict) and "model_state_dict" in state:
            m.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            m.load_state_dict(state["state_dict"])
        else:
            m.load_state_dict(state)   # raw OrderedDict

        m.to(device)
        m.eval()
        print(f"[✓] Model loaded successfully  →  {MODEL_PATH}")
        print(f"[✓] Running on: {device}")
        return m

    except FileNotFoundError:
        print(f"[✗] Model file not found: {MODEL_PATH}")
        return None
    except RuntimeError as e:
        print(f"[✗] Architecture mismatch while loading model: {e}")
        return None
    except Exception as e:
        print(f"[✗] Unexpected error loading model: {e}")
        return None


model = load_model()


# ─────────────────────────────────────────────────────────────
#  Image Preprocessing  (224×224, ImageNet normalisation)
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(image_path: str) -> torch.Tensor:
    """Open image, convert to RGB, apply transforms, add batch dim."""
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────
def predict(image_path: str):
    """
    Run inference and return (class_name, confidence_percent).
    Returns (None, None) if model failed to load.
    """
    if model is None:
        return None, None

    tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs      = model(tensor)
        probs        = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    class_name     = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    return class_name, confidence_pct


# ─────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Home / Upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """Receive uploaded image → preprocess → predict → render result."""

    # 1. File present in request?
    if "file" not in request.files:
        flash("No file part in the request. Please select an image.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    # 2. File actually chosen?
    if file.filename == "":
        flash("No image selected. Please upload a flower image.", "error")
        return redirect(url_for("index"))

    # 3. Valid extension?
    if not allowed_file(file.filename):
        flash(
            "Unsupported file format. Please upload a JPG, JPEG, or PNG image.",
            "error",
        )
        return redirect(url_for("index"))

    try:
        # 4. Save securely
        filename  = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # 5. Verify it's a real image (catches corrupted files)
        try:
            with Image.open(save_path) as img:
                img.verify()
        except Exception:
            if os.path.exists(save_path):
                os.remove(save_path)
            flash("The uploaded file is corrupted or not a valid image.", "error")
            return redirect(url_for("index"))

        # 6. Model loaded?
        if model is None:
            flash(
                "Model could not be loaded. "
                "Ensure flower_classification_model.pth is inside the model/ folder.",
                "error",
            )
            return redirect(url_for("index"))

        # 7. Predict
        class_name, confidence = predict(save_path)

        if class_name is None:
            flash("Prediction failed. Please try a different image.", "error")
            return redirect(url_for("index"))

        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            image_url=image_url,
            class_name=class_name.title(),
            confidence=confidence,
        )

    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
