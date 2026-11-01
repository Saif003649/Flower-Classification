# 🌸 Flower Classification Using Deep Learning

A modern web-based flower classification system built with **PyTorch**, **Hugging Face Transformers**, and **Flask**.  
Upload any flower photograph and classify it across **102 Oxford Flower species** using your choice of a state-of-the-art **Vision Transformer (ViT)** or a lightweight **Custom CNN**.

---

## 📸 Key Features

* **Dual Model Architecture:**
  * **⚡ Vision Transformer (ViT-Base):** `loretyan/vit-base-oxford-flowers-102` fine-tuned on the Oxford 102 dataset. Uses self-attention mechanisms across image patches with verified semantic label decoding. Achieves **99.72% Top-1 accuracy** on the official Oxford 102 test partition (6,149 images).
  * **🧠 Custom CNN (Baseline):** 2-block convolutional neural network (`flower_classification_model.pth`) providing an efficient, native PyTorch baseline with ultra-fast CPU latency (~6.8 ms).
* **Enhanced Modern UI/UX (Phase 7):**
  * **Intuitive Model Selector:** Interactive radio card selector with accuracy & speed indicators.
  * **Drag & Drop Upload:** Fluid drag-over animations, instant image preview, aspect-ratio containment, file metadata (name, formatted size), and clear/remove options.
  * **Client-Side Validation & Security:** Validates file size (≤ 16 MB) and formats (JPG, JPEG, PNG) with friendly inline alerts. Never leaks internal server file paths or stack traces.
  * **Loading & Inference States:** Animated dual-ring petal spinner with dynamic status reflecting the selected model architecture and back/forward cache restore.
  * **Rich Prediction Cards:** Primary identified species with botanical styling, primary confidence bar, and confidence level interpretation (High, Moderate, Ambiguous).
  * **Top-3 Candidates with Progress Bars:** Visual ranked cards (#1, #2, #3) with individual confidence progress bars and percentage metrics.
  * **Browser Session History:** Recent predictions stored strictly in `sessionStorage` (cleared on browser tab close; zero permanent server disk retention) with quick-clear option.
  * **Fully Responsive:** Fluid layouts optimized for desktop (2-column grid), tablet, and mobile (touch-friendly 44px+ tap targets).
* **Robust Backend:**
  * In-memory model caching (models load once at startup, not per request).
  * Strict evaluation mode (`model.eval()` and `torch.inference_mode()`).
  * Image validation and graceful error handling for corrupt files or inference failures.

---

## 📊 Model Benchmark Comparison

Evaluated on the official **Oxford Flowers 102** test partition (**6,149 images**):

| Metric | Vision Transformer (ViT-Base) | Custom CNN (PyTorch Baseline) |
|:---|:---:|:---:|
| **Architecture** | 12-layer Transformer (Self-Attention) | 2-block ConvNet + MLP |
| **Parameters** | 86,567,642 (~86.6 M) | 12,863,430 (~12.9 M) |
| **Checkpoint Size** | ~328 MB (Hugging Face) | 49.07 MB (`.pth`) |
| **Top-1 Accuracy** | **99.72%** | **5.68%** |
| **Top-5 Accuracy** | **100.00%** | **10.77%** |
| **Macro-F1 Score** | **99.74%** | **7.84%** |
| **CPU Latency** | ~213 ms / image | **~6.76 ms / image** (~31× faster) |
| **Optimal Use Case** | High-precision botanical identification | Lightweight real-time edge/CPU baseline |

---

## 🧰 Tech Stack

| Component | Technology | Description |
|:---|:---|:---|
| **Language** | Python 3.11 / 3.13 | Core backend language |
| **Deep Learning** | PyTorch & Torchvision | Neural network tensors, transforms, and execution |
| **Transformer Backbone** | Hugging Face `transformers` | Vision Transformer architecture & preprocessors |
| **Web Server** | Flask 3.x | Lightweight WSGI application & REST routing |
| **Image Processing** | Pillow (PIL) | Format verification, RGB conversion, and resizing |
| **Frontend UI** | HTML5, Modern CSS3, Vanilla JS | Custom responsive UI with Jinja2 templating |
| **Session Storage** | HTML5 `sessionStorage` | In-browser transient prediction history |
| **Evaluation Metrics** | scikit-learn | Accuracy, Top-5 accuracy, Macro-F1, Confusion Matrix |

---

## 📁 Project Directory Structure

```
flower-classification/
├── app.py                             # Main Flask application server
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusion rules (datasets, cache, uploads)
├── benchmark_models.py                # Comparative benchmarking script (CNN vs ViT)
├── benchmark_results.json             # Official benchmark metrics output
├── benchmark_comparison.csv           # Tabular side-by-side benchmark comparison
├── evaluate_oxford_vit.py             # Benchmark script for ViT on Oxford 102 test set
├── oxford_vit_evaluation_metrics.json # Pre-computed ViT evaluation metrics
├── oxford_vit_confusion_matrix.csv    # 102×102 ViT confusion matrix
├── test_model.py                      # CLI testing script for both CNN & ViT
├── test_integration.py                # End-to-end integration test suite
├── test_phase7_ui.py                  # Phase 7 UI/UX and security verification suite
│
├── model/
│   └── flower_classification_model.pth # Custom CNN PyTorch checkpoint (102 classes)
│
├── static/
│   ├── css/
│   │   └── style.css                  # Application styles, design tokens & responsive rules
│   ├── js/
│   │   └── script.js                  # Drag & drop, preview, loading state & session history
│   └── uploads/                       # Temporary upload staging directory (.gitkeep tracked)
│
└── templates/
    ├── base.html                      # Common header, navigation, and footer
    ├── index.html                     # Upload page with model selector & session history
    ├── result.html                    # Detailed prediction card with Top-3 progress bars
    └── about.html                     # Model specifications, benchmark table & disclaimers
```

---

## ⚙️ Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone <repository-url>
cd flower-classification
```

### 2. (Optional) Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Start the Flask development server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

1. Select your model: **Vision Transformer (ViT-Base)** *(Recommended)* or **Custom CNN**.
2. Drag and drop or browse for a flower image (JPG, JPEG, or PNG).
3. Click **Classify Flower** to view the top classification, confidence score, and top-3 candidates with progress bars.

---

## 🧪 Testing & Verification Suites

### 1. Run UI/UX & Security Verification Suite
```bash
python test_phase7_ui.py
```
*Verifies: Home page components, About benchmark table, friendly error handling without path leaks, ViT and CNN workflows, Top-3 progress bars, session history recording, and model checkpoint integrity.*

### 2. Run End-to-End Integration Suite
```bash
python test_integration.py
```

### 3. Test Both Models on Sample Images (CLI)
```bash
# Test both models side-by-side
python test_model.py

# Test only the Vision Transformer
python test_model.py --model vit

# Test only the Custom CNN
python test_model.py --model cnn
```

### 4. Run Model Benchmark (CNN vs ViT)
```bash
# Quick mode (evaluates CNN, reuses cached ViT metrics)
python benchmark_models.py --skip-vit-eval

# Full mode (re-evaluates both models across 6,149 test images)
python benchmark_models.py
```

---

## ℹ️ Academic & Real-World Notice

* Benchmark metrics (such as the 99.72% test accuracy for ViT on the Oxford 102 dataset) reflect performance on curated, standardized evaluation images.
* Real-world garden photos may exhibit varied lighting, heavy occlusion, motion blur, or rare flower sub-varieties not represented in the training set. Model confidence scores are provided to help assess prediction certainty.
* User uploads are processed transiently and are not stored permanently. Neither raw dataset partitions nor user uploads are tracked in version control (configured in `.gitignore`).
