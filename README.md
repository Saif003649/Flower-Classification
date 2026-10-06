# 🌸 Flower Classification Using Deep Learning

A web-based flower classification application built with **PyTorch** and **Flask**.  
Upload any flower image and the deep learning model will identify it instantly with a confidence score.

---

## 📸 What It Does

1. User opens the website and uploads a flower image (JPG / JPEG / PNG).
2. The image is sent to the Flask backend.
3. Flask preprocesses the image (resize → normalize → tensor).
4. The pretrained PyTorch model predicts the flower class.
5. The result page displays:
   - The uploaded image
   - Predicted flower name
   - Model confidence percentage

---

## 🧰 Technologies Used

| Technology    | Purpose                          |
|---------------|----------------------------------|
| Python 3.9+   | Backend language                 |
| PyTorch       | Deep learning inference          |
| Torchvision   | Model architecture + transforms  |
| Flask         | Web framework / server           |
| Pillow (PIL)  | Image loading & preprocessing    |
| HTML5 / CSS3  | Frontend UI                      |
| JavaScript    | Image preview & drag-drop        |
| Jinja2        | HTML templating                  |

---

## 📁 Project Structure

```
FlowerClassification/
│
├── app.py                   ← Main Flask application
│
├── model/
│   └── flower_model.pth     ← Pretrained PyTorch model (place here)
│
├── static/
│   ├── css/
│   │   └── style.css        ← All styles (beige theme)
│   ├── js/
│   │   └── script.js        ← Image preview & drag-drop
│   └── uploads/             ← Uploaded images saved here
│
├── templates/
│   ├── base.html            ← Shared layout (navbar, footer)
│   ├── index.html           ← Home / upload page
│   ├── result.html          ← Prediction result page
│   └── about.html           ← About page
│
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

---

## ⚙️ Installation & Setup

### 1. Clone / Download the Project

```bash
cd e:\Flower
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If you have a GPU, install PyTorch with CUDA support from https://pytorch.org/get-started/locally/

### 5. Place the Model File

Copy your pretrained `.pth` file to:
```
model/flower_model.pth
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open the App

Open your browser and go to:
```
http://localhost:5000
```

---

## 🔧 Configuration

### Changing Flower Classes

Open `app.py` and edit the `CLASS_NAMES` list to match your model's output:

```python
CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
]
```

Make sure the **order matches the order used during training**.

### Changing Model Architecture

By default, the app uses **ResNet18** with a modified final layer.  
If your model uses a different architecture, update the `build_model()` function in `app.py`.

### Changing Image Size

If your model expects a different input size (e.g., 128×128 or 256×256), update:
```python
IMAGE_SIZE = 224   # Change this
```

---

## 🌐 How the Pipeline Works

```
User uploads image
       ↓
Flask receives file
       ↓
Validate extension (JPG/JPEG/PNG)
       ↓
Save to static/uploads/
       ↓
PIL opens & converts to RGB
       ↓
Resize to 224×224
       ↓
Normalize (ImageNet mean/std)
       ↓
Convert to PyTorch Tensor
       ↓
Add batch dimension [1, 3, 224, 224]
       ↓
Pass to model.eval() with torch.no_grad()
       ↓
Apply Softmax → get probabilities
       ↓
Find max probability → class name
       ↓
Render result.html with prediction
```

---

## 🎓 Viva / Exam Key Points

1. **Why PyTorch?** — Industry-standard deep learning framework with dynamic computation graphs.
2. **What is Transfer Learning?** — Using a model pretrained on large data (ImageNet) and fine-tuning it for a smaller specific task.
3. **Why Softmax?** — Converts raw model output (logits) into probabilities that sum to 1.
4. **Why torch.no_grad()?** — Disables gradient tracking during inference to save memory and speed up prediction.
5. **Why model.eval()?** — Switches off dropout and batch normalization training behavior.
6. **What is normalization?** — Scaling pixel values using mean/std so input matches training distribution.
7. **What is secure_filename?** — Prevents directory traversal attacks from malicious filenames.
8. **Why load model once?** — Loading model on each request is extremely slow; loading once at startup is efficient.
9. **What is Jinja2?** — Python templating engine used by Flask to inject variables into HTML.
10. **What is Flask?** — Lightweight Python web framework for building web applications.

---

## ⚠️ Disclaimer

This is an academic deep learning project.  
Predictions are based on the trained model and may not be accurate for all real-world images.

---

## 📄 License

For educational use only.
