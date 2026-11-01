"""
Integration & End-to-End Test Suite for Flower Classification System
Tests:
1. Model loading & preprocessing verification for both CNN and ViT
2. Label decoding verification (CNN CLASS_NAMES vs ViT id2label)
3. Inference & top-3 prediction structure for both models
4. Flask route tests via Flask test client:
   - GET / (Home page, model options present)
   - GET /about (About page, both architectures documented)
   - POST /predict with missing file (error flash & redirect)
   - POST /predict with invalid extension (error flash & redirect)
   - POST /predict with corrupted file (error flash & redirect)
   - POST /predict with valid image & model_choice='vit' (200, result page, top-3 candidates)
   - POST /predict with valid image & model_choice='cnn' (200, result page, top-3 candidates)
"""

import os
import sys
import io
from PIL import Image

# Import app components
import app as flask_app
import torch

def test_models_loaded():
    """Verify both models are loaded in memory."""
    assert flask_app.cnn_model is not None, "CNN model failed to load at startup"
    assert flask_app.vit_model is not None, "ViT model failed to load at startup"
    assert flask_app.vit_processor is not None, "ViT processor failed to load at startup"
    assert len(flask_app.vit_id2label) == 102, f"ViT id2label should have 102 classes, got {len(flask_app.vit_id2label)}"
    assert len(flask_app.CLASS_NAMES) == 102, f"CNN CLASS_NAMES should have 102 classes, got {len(flask_app.CLASS_NAMES)}"

def test_preprocessing_separation():
    """Verify CNN and ViT use distinct, appropriate preprocessing pipelines."""
    # Create test synthetic image
    test_img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    temp_path = os.path.join(flask_app.UPLOAD_FOLDER, "temp_test_pipe.png")
    test_img.save(temp_path)

    try:
        # 1. Test CNN preprocessing
        img_open = Image.open(temp_path).convert("RGB")
        cnn_tensor = flask_app.cnn_transform(img_open).unsqueeze(0)
        assert cnn_tensor.shape == (1, 3, 224, 224), f"CNN tensor shape mismatch: {cnn_tensor.shape}"
        assert cnn_tensor.dtype == torch.float32

        # 2. Test ViT preprocessing
        vit_inputs = flask_app.vit_processor(images=img_open, return_tensors="pt")
        assert "pixel_values" in vit_inputs, "ViT processor must output pixel_values"
        assert vit_inputs["pixel_values"].shape == (1, 3, 224, 224), f"ViT pixel_values shape mismatch: {vit_inputs['pixel_values'].shape}"
        assert vit_inputs["pixel_values"].dtype == torch.float32
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_label_decoding_separation():
    """Verify ViT uses vit_id2label and CNN uses CLASS_NAMES."""
    # Check balloon flower index
    # In CNN CLASS_NAMES: index 18 is 'balloon flower'
    assert flask_app.CLASS_NAMES[18] == "balloon flower"
    # In ViT id2label: index 19 is 'balloon flower' (due to dataset cyclic rotation)
    assert flask_app.vit_id2label[19] == "balloon flower"
    # Check index 40
    assert flask_app.CLASS_NAMES[40] == "barberton daisy"
    assert flask_app.vit_id2label[40] == "barberton daisy"

def test_inference_top3_output():
    """Verify predict_cnn and predict_vit return 3 sorted candidates with class_name and confidence."""
    # Use an existing test image in static/uploads
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = [
        f for f in sorted(os.listdir(flask_app.UPLOAD_FOLDER))
        if os.path.splitext(f)[1].lower() in valid_exts and not f.startswith("temp_")
    ]
    assert len(images) > 0, "At least one test image required in static/uploads"

    sample_img_path = os.path.join(flask_app.UPLOAD_FOLDER, images[0])

    # Test CNN prediction
    cnn_top3 = flask_app.predict_cnn(sample_img_path)
    assert len(cnn_top3) == 3, f"CNN should return 3 candidates, got {len(cnn_top3)}"
    assert all("class_name" in c and "confidence" in c for c in cnn_top3)
    assert cnn_top3[0]["confidence"] >= cnn_top3[1]["confidence"] >= cnn_top3[2]["confidence"]

    # Test ViT prediction
    vit_top3 = flask_app.predict_vit(sample_img_path)
    assert len(vit_top3) == 3, f"ViT should return 3 candidates, got {len(vit_top3)}"
    assert all("class_name" in c and "confidence" in c for c in vit_top3)
    assert vit_top3[0]["confidence"] >= vit_top3[1]["confidence"] >= vit_top3[2]["confidence"]

def test_flask_routes():
    """Test Flask client endpoints."""
    app = flask_app.app
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. GET /
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Vision Transformer (ViT-Base)" in html
    assert "Custom CNN (PyTorch)" in html
    assert 'name="model_choice"' in html

    # 2. GET /about
    res_about = client.get("/about")
    assert res_about.status_code == 200
    about_html = res_about.get_data(as_text=True)
    assert "Vision Transformer" in about_html
    assert "Custom CNN" in about_html
    assert "loretyan/vit-base-oxford-flowers-102" in about_html

    # 3. POST /predict without file
    res_nofile = client.post("/predict", data={}, follow_redirects=True)
    assert res_nofile.status_code == 200
    assert "No file part in the request" in res_nofile.get_data(as_text=True)

    # 4. POST /predict with invalid extension (.txt)
    data_invalid_ext = {
        "file": (io.BytesIO(b"fake text content"), "test.txt"),
        "model_choice": "vit"
    }
    res_inval = client.post("/predict", data=data_invalid_ext, content_type="multipart/form-data", follow_redirects=True)
    assert res_inval.status_code == 200
    assert "Unsupported file format" in res_inval.get_data(as_text=True)

    # 5. POST /predict with corrupt image (.png with garbage bytes)
    data_corrupt = {
        "file": (io.BytesIO(b"\x89PNG\r\n\x1a\ncorrupted data that is not a real image"), "corrupted.png"),
        "model_choice": "vit"
    }
    res_corrupt = client.post("/predict", data=data_corrupt, content_type="multipart/form-data", follow_redirects=True)
    assert res_corrupt.status_code == 200
    assert "corrupted or not a valid image" in res_corrupt.get_data(as_text=True)

    # 6. POST /predict with valid image (ViT model choice)
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (224, 224), color=(180, 80, 120)).save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    data_valid_vit = {
        "file": (img_byte_arr, "valid_flower.png"),
        "model_choice": "vit"
    }
    res_vit = client.post("/predict", data=data_valid_vit, content_type="multipart/form-data", follow_redirects=True)
    assert res_vit.status_code == 200
    vit_html = res_vit.get_data(as_text=True)
    assert "Vision Transformer (ViT-Base)" in vit_html
    assert "Top Prediction" in vit_html
    assert "Top 3 Candidates" in vit_html
    assert "#1" in vit_html and "#2" in vit_html and "#3" in vit_html

    # 7. POST /predict with valid image (CNN model choice)
    img_byte_arr_cnn = io.BytesIO()
    Image.new("RGB", (224, 224), color=(80, 180, 120)).save(img_byte_arr_cnn, format="PNG")
    img_byte_arr_cnn.seek(0)
    data_valid_cnn = {
        "file": (img_byte_arr_cnn, "valid_flower_cnn.png"),
        "model_choice": "cnn"
    }
    res_cnn = client.post("/predict", data=data_valid_cnn, content_type="multipart/form-data", follow_redirects=True)
    assert res_cnn.status_code == 200
    cnn_html = res_cnn.get_data(as_text=True)
    assert "Custom CNN (PyTorch)" in cnn_html
    assert "Top Prediction" in cnn_html
    assert "Top 3 Candidates" in cnn_html
    assert "#1" in cnn_html and "#2" in cnn_html and "#3" in cnn_html

if __name__ == "__main__":
    print("Running integration tests...")
    test_models_loaded()
    print("[OK] Model loading verified.")
    test_preprocessing_separation()
    print("[OK] Preprocessing pipeline separation verified.")
    test_label_decoding_separation()
    print("[OK] Label decoding separation verified.")
    test_inference_top3_output()
    print("[OK] Inference & Top-3 output format verified.")
    test_flask_routes()
    print("[OK] Flask routes, error handling, model switching verified.")
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
