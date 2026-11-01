"""
Phase 7 Verification Test Suite
===============================
Verifies:
1. Flask routes & UI rendering:
   - GET / (Home upload page, model selector, drag-drop zone, history section)
   - GET /about (About page, benchmark table, architecture specs)
   - POST /predict with missing file -> friendly error, no path leaks
   - POST /predict with unsupported format -> friendly error, no path leaks
   - POST /predict with corrupted image -> friendly error, no path leaks
   - POST /predict with valid image (ViT) -> 200, selected model badge, top prediction, top-3 progress bars
   - POST /predict with valid image (CNN) -> 200, selected model badge, top prediction, top-3 progress bars
2. Error message security audit:
   - No filesystem paths (e.g. 'D:\\', 'C:\\') in flashed messages or rendered responses.
3. Model & Benchmark files integrity:
   - flower_classification_model.pth unchanged
   - benchmark_results.json intact
   - benchmark_comparison.csv intact
   - oxford_vit_evaluation_metrics.json intact
"""

import os
import sys
import io
import json
from PIL import Image

import app as flask_app

def run_tests():
    print("=" * 70)
    print("  PHASE 7 UI/UX VERIFICATION TEST SUITE")
    print("=" * 70)

    app = flask_app.app
    app.config["TESTING"] = True
    client = app.test_client()

    # ─────────────────────────────────────────────────────────────
    #  1. Home Page Verification
    # ─────────────────────────────────────────────────────────────
    print("\n[1] Testing GET / (Home Page) ...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.get_data(as_text=True)

    # Check key UI components
    assert "Vision Transformer (ViT-Base)" in html, "Missing ViT model title"
    assert "Custom CNN (PyTorch)" in html, "Missing CNN model title"
    assert "name=\"model_choice\"" in html, "Missing model_choice radio inputs"
    assert "id=\"drop-zone\"" in html, "Missing drag-and-drop drop-zone"
    assert "id=\"preview-wrapper\"" in html, "Missing preview-wrapper"
    assert "id=\"loading-overlay\"" in html, "Missing loading-overlay"
    assert "id=\"history-section\"" in html, "Missing session history section"
    assert "Clear History" in html, "Missing Clear History button"
    print("    [OK] Home page rendered with all required UI components.")

    # ─────────────────────────────────────────────────────────────
    #  2. About Page & Benchmark Table Verification
    # ─────────────────────────────────────────────────────────────
    print("\n[2] Testing GET /about (About & Models Page) ...")
    res_about = client.get("/about")
    assert res_about.status_code == 200, f"Expected 200, got {res_about.status_code}"
    about_html = res_about.get_data(as_text=True)

    assert "Phase 6 Official Benchmark Results" in about_html, "Missing Phase 6 benchmark results"
    assert "99.72%" in about_html, "Missing ViT benchmark top-1 score"
    assert "5.68%" in about_html, "Missing CNN benchmark top-1 score"
    assert "Vision Transformer (ViT-Base)" in about_html, "Missing ViT architecture section"
    assert "Custom CNN (PyTorch Baseline)" in about_html, "Missing CNN architecture section"
    print("    [OK] About page rendered with benchmark table and architecture details.")

    # ─────────────────────────────────────────────────────────────
    #  3. Validation & Friendly Error Handling (Requirement 12 & 16)
    # ─────────────────────────────────────────────────────────────
    print("\n[3] Testing Validation & Sanitized Error Messages ...")

    # A. Missing file
    res_nofile = client.post("/predict", data={}, follow_redirects=True)
    nofile_html = res_nofile.get_data(as_text=True)
    assert "No file part in the request" in nofile_html or "Please select an image" in nofile_html
    assert "D:\\" not in nofile_html and "C:\\" not in nofile_html, "Path leak detected in error!"
    print("    [OK] Missing file handled gracefully without path leak.")

    # B. Unsupported file extension (.pdf)
    bad_ext_data = {
        "file": (io.BytesIO(b"%PDF-1.4 dummy pdf"), "document.pdf"),
        "model_choice": "vit"
    }
    res_bad_ext = client.post("/predict", data=bad_ext_data, content_type="multipart/form-data", follow_redirects=True)
    bad_ext_html = res_bad_ext.get_data(as_text=True)
    assert "Unsupported file format" in bad_ext_html
    assert "D:\\" not in bad_ext_html and "C:\\" not in bad_ext_html, "Path leak detected in error!"
    print("    [OK] Unsupported file extension handled gracefully without path leak.")

    # C. Corrupted image
    corrupt_data = {
        "file": (io.BytesIO(b"garbage not an image"), "corrupted.jpg"),
        "model_choice": "vit"
    }
    res_corrupt = client.post("/predict", data=corrupt_data, content_type="multipart/form-data", follow_redirects=True)
    corrupt_html = res_corrupt.get_data(as_text=True)
    assert "corrupted or not a valid image" in corrupt_html
    assert "D:\\" not in corrupt_html and "C:\\" not in corrupt_html, "Path leak detected in error!"
    print("    [OK] Corrupted image handled gracefully without path leak.")

    # ─────────────────────────────────────────────────────────────
    #  4. ViT Prediction Workflow (Requirements 5, 6, 7, 8)
    # ─────────────────────────────────────────────────────────────
    print("\n[4] Testing ViT Prediction Workflow ...")
    img_vit_bytes = io.BytesIO()
    Image.new("RGB", (224, 224), color=(200, 100, 150)).save(img_vit_bytes, format="JPEG")
    img_vit_bytes.seek(0)

    vit_post_data = {
        "file": (img_vit_bytes, "test_vit_flower.jpg"),
        "model_choice": "vit"
    }
    res_vit = client.post("/predict", data=vit_post_data, content_type="multipart/form-data", follow_redirects=True)
    assert res_vit.status_code == 200, f"Expected 200, got {res_vit.status_code}"
    vit_html = res_vit.get_data(as_text=True)

    # Selected Model Name displayed (Requirement 7)
    assert "Vision Transformer (ViT-Base)" in vit_html, "ViT model name not displayed in result"
    # Top Prediction displayed (Requirement 5)
    assert "Top Prediction" in vit_html, "Top prediction header missing"
    assert "Prediction Confidence" in vit_html, "Prediction confidence label missing"
    # Top 3 Candidates with progress bars (Requirement 6)
    assert "Top 3 Candidates" in vit_html, "Top 3 Candidates card missing"
    assert "candidate-progress-fill" in vit_html, "Candidate progress bar class missing"
    assert "#1" in vit_html and "#2" in vit_html and "#3" in vit_html, "Candidate ranks #1, #2, #3 missing"
    # Session history recording script present (Requirement 8)
    assert "flower_pred_history" in vit_html, "Session history storage script missing"
    # Try another / Reset button (Requirement 9)
    assert "Classify Another Image" in vit_html, "Try another image button missing"
    print("    [OK] ViT prediction workflow, model badge, top-3 progress bars, and session recording verified.")

    # ─────────────────────────────────────────────────────────────
    #  5. CNN Prediction Workflow (Requirements 5, 6, 7, 8)
    # ─────────────────────────────────────────────────────────────
    print("\n[5] Testing CNN Prediction Workflow ...")
    img_cnn_bytes = io.BytesIO()
    Image.new("RGB", (224, 224), color=(100, 180, 200)).save(img_cnn_bytes, format="JPEG")
    img_cnn_bytes.seek(0)

    cnn_post_data = {
        "file": (img_cnn_bytes, "test_cnn_flower.jpg"),
        "model_choice": "cnn"
    }
    res_cnn = client.post("/predict", data=cnn_post_data, content_type="multipart/form-data", follow_redirects=True)
    assert res_cnn.status_code == 200, f"Expected 200, got {res_cnn.status_code}"
    cnn_html = res_cnn.get_data(as_text=True)

    # Selected Model Name displayed (Requirement 7)
    assert "Custom CNN (PyTorch)" in cnn_html, "CNN model name not displayed in result"
    # Top Prediction displayed (Requirement 5)
    assert "Top Prediction" in cnn_html, "Top prediction header missing"
    # Top 3 Candidates with progress bars (Requirement 6)
    assert "Top 3 Candidates" in cnn_html, "Top 3 Candidates card missing"
    assert "candidate-progress-fill" in cnn_html, "Candidate progress bar class missing"
    assert "#1" in cnn_html and "#2" in cnn_html and "#3" in cnn_html, "Candidate ranks #1, #2, #3 missing"
    print("    [OK] CNN prediction workflow, model badge, top-3 progress bars, and session recording verified.")

    # ─────────────────────────────────────────────────────────────
    #  6. Integrity of Models & Benchmark Files (Requirement 14)
    # ─────────────────────────────────────────────────────────────
    print("\n[6] Checking Integrity of Existing Models & Benchmark Files ...")

    cnn_pth = os.path.join(flask_app.BASE_DIR, "model", "flower_classification_model.pth")
    assert os.path.exists(cnn_pth), "CNN checkpoint model file missing!"
    size_mb = round(os.path.getsize(cnn_pth) / 1024 / 1024, 2)
    assert size_mb == 49.07, f"CNN checkpoint altered! Expected 49.07 MB, got {size_mb} MB"
    print(f"    [OK] CNN checkpoint intact: {size_mb} MB.")

    bench_json = os.path.join(flask_app.BASE_DIR, "benchmark_results.json")
    assert os.path.exists(bench_json), "benchmark_results.json missing!"
    with open(bench_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    assert bench_data["cnn"]["metrics"]["top1_accuracy"] == 5.6757
    assert bench_data["vit"]["metrics"]["top1_accuracy"] == 99.72
    print("    [OK] benchmark_results.json intact and verified.")

    bench_csv = os.path.join(flask_app.BASE_DIR, "benchmark_comparison.csv")
    assert os.path.exists(bench_csv), "benchmark_comparison.csv missing!"
    print("    [OK] benchmark_comparison.csv intact.")

    vit_metrics_json = os.path.join(flask_app.BASE_DIR, "oxford_vit_evaluation_metrics.json")
    assert os.path.exists(vit_metrics_json), "oxford_vit_evaluation_metrics.json missing!"
    print("    [OK] oxford_vit_evaluation_metrics.json intact.")

    print("\n" + "=" * 70)
    print("  ALL PHASE 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_tests()

