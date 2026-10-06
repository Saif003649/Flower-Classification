/**
 * Flower Classification App — Main JavaScript
 * Handles: image preview, drag-drop, form submit loading overlay
 */

(function () {
  "use strict";

  // ─── DOM References ──────────────────────────────────────────
  const dropZone       = document.getElementById("drop-zone");
  const fileInput      = document.getElementById("file-input");
  const dropContent    = document.getElementById("drop-zone-content");
  const previewWrapper = document.getElementById("preview-wrapper");
  const previewImage   = document.getElementById("preview-image");
  const previewName    = document.getElementById("preview-name");
  const removeBtn      = document.getElementById("remove-btn");
  const classifyBtn    = document.getElementById("classify-btn");
  const uploadForm     = document.getElementById("upload-form");
  const loadingOverlay = document.getElementById("loading-overlay");

  // Only run if we're on the upload page
  if (!dropZone) return;

  // ─── Show Image Preview ───────────────────────────────────────
  function showPreview(file) {
    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png"];
    if (!allowedTypes.includes(file.type)) {
      alert("Unsupported file format. Please upload a JPG, JPEG, or PNG image.");
      resetUI();
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src    = e.target.result;
      previewName.textContent = file.name;
      dropContent.style.display  = "none";
      previewWrapper.style.display = "flex";
      classifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // ─── Reset Drop Zone ─────────────────────────────────────────
  function resetUI() {
    previewImage.src    = "#";
    previewName.textContent = "";
    dropContent.style.display  = "flex";
    previewWrapper.style.display = "none";
    classifyBtn.disabled = true;
    fileInput.value = "";
  }

  // ─── File Input Change ────────────────────────────────────────
  fileInput.addEventListener("change", function () {
    if (this.files && this.files[0]) {
      showPreview(this.files[0]);
    }
  });

  // ─── Click on drop zone opens file dialog ────────────────────
  dropZone.addEventListener("click", function (e) {
    // Don't re-trigger if clicking the Browse label or Remove button
    if (
      e.target.tagName === "LABEL" ||
      e.target === removeBtn ||
      e.target.tagName === "INPUT"
    ) return;
    fileInput.click();
  });

  // ─── Remove Button ────────────────────────────────────────────
  removeBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    resetUI();
  });

  // ─── Drag & Drop ─────────────────────────────────────────────
  dropZone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
      // Assign to file input so the form can submit it
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      showPreview(file);
    }
  });

  // ─── Form Submit → Show Loading Overlay ──────────────────────
  uploadForm.addEventListener("submit", function () {
    if (loadingOverlay) {
      loadingOverlay.style.display = "flex";
    }
  });

})();
