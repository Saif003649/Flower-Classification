/**
 * Flower Classification System — Enhanced UI/UX JavaScript (Phase 7)
 * Handles:
 *  - Drag & drop image upload with visual feedback
 *  - Instant image preview & file metadata display
 *  - Client-side validation (types, size limit <= 16MB) with friendly error alerts
 *  - Intuitive model selector interaction & dynamic loading status
 *  - Clear & reset controls for upload and session history
 *  - Prediction loading overlay with back-forward cache support
 *  - Browser session prediction history (sessionStorage only, no permanent server storage)
 */

(function () {
  "use strict";

  // ─────────────────────────────────────────────────────────────
  //  Constants & Configuration
  // ─────────────────────────────────────────────────────────────
  const MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB
  const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"];
  const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"];
  const STORAGE_KEY = "flower_pred_history";

  // ─────────────────────────────────────────────────────────────
  //  DOM Element Selectors
  // ─────────────────────────────────────────────────────────────
  const dropZone          = document.getElementById("drop-zone");
  const fileInput         = document.getElementById("file-input");
  const dropContent       = document.getElementById("drop-zone-content");
  const previewWrapper    = document.getElementById("preview-wrapper");
  const previewImage      = document.getElementById("preview-image");
  const previewName       = document.getElementById("preview-name");
  const previewSize       = document.getElementById("preview-size");
  const previewType       = document.getElementById("preview-type");
  const removeBtn         = document.getElementById("remove-btn");
  const resetAllBtn       = document.getElementById("reset-all-btn");
  const classifyBtn       = document.getElementById("classify-btn");
  const classifyBtnText   = document.getElementById("classify-btn-text");
  const uploadForm        = document.getElementById("upload-form");
  const loadingOverlay    = document.getElementById("loading-overlay");
  const loadingTitle      = document.getElementById("loading-title");
  const loadingSubtitle   = document.getElementById("loading-subtitle");
  const stepStatus        = document.getElementById("step-status");
  const errorBanner       = document.getElementById("client-error-banner");
  const errorText         = document.getElementById("client-error-text");
  const errorClose        = document.getElementById("client-error-close");

  // History DOM
  const historySection    = document.getElementById("history-section");
  const historyGrid       = document.getElementById("history-grid");
  const historyCount      = document.getElementById("history-count");
  const clearHistoryBtn   = document.getElementById("clear-history-btn");

  // Model Selection DOM
  const modelRadios       = document.querySelectorAll('input[name="model_choice"]');
  const modelOptions      = document.querySelectorAll('.model-option');

  // Prevent default drag & drop behaviors on the whole document
  window.addEventListener("dragover", function (e) { e.preventDefault(); }, false);
  window.addEventListener("drop", function (e) { e.preventDefault(); }, false);

  // ─────────────────────────────────────────────────────────────
  //  Client Error Alert Banner
  // ─────────────────────────────────────────────────────────────
  function showError(message) {
    if (!errorBanner || !errorText) {
      alert(message);
      return;
    }
    errorText.textContent = message;
    errorBanner.style.display = "flex";
    errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.style.display = "none";
      if (errorText) errorText.textContent = "";
    }
  }

  if (errorClose) {
    errorClose.addEventListener("click", hideError);
  }

  // ─────────────────────────────────────────────────────────────
  //  Format File Size
  // ─────────────────────────────────────────────────────────────
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(1) + " " + sizes[i];
  }

  // ─────────────────────────────────────────────────────────────
  //  Validate File
  // ─────────────────────────────────────────────────────────────
  function validateFile(file) {
    hideError();

    if (!file) {
      showError("Please select a flower image to classify.");
      return false;
    }

    // 1. File size check
    if (file.size > MAX_FILE_SIZE_BYTES) {
      showError(`The selected file is too large (${formatBytes(file.size)}). Maximum allowed size is 16 MB.`);
      return false;
    }

    // 2. Extension check
    const filename = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTENSIONS.some(function (ext) {
      return filename.endsWith(ext);
    });

    // 3. MIME type check
    const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type.toLowerCase()) || file.type === "";

    if (!hasValidExt && !hasValidMime) {
      showError("Unsupported file format. Please upload a valid JPG, JPEG, or PNG photograph.");
      return false;
    }

    return true;
  }

  // ─────────────────────────────────────────────────────────────
  //  Image Preview Display
  // ─────────────────────────────────────────────────────────────
  function showPreview(file) {
    if (!validateFile(file)) {
      clearPreview();
      return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
      if (previewImage) {
        previewImage.src = e.target.result;
      }
      if (previewName) {
        previewName.textContent = file.name;
      }
      if (previewSize) {
        previewSize.textContent = formatBytes(file.size);
      }
      if (previewType) {
        const ext = file.name.split(".").pop().toUpperCase();
        previewType.textContent = ext ? `${ext} Photo` : "Image";
      }

      if (dropContent) dropContent.style.display = "none";
      if (previewWrapper) previewWrapper.style.display = "flex";
      if (classifyBtn) classifyBtn.disabled = false;
      if (resetAllBtn) resetAllBtn.style.display = "inline-flex";

      if (stepStatus) {
        stepStatus.textContent = "Photo ready";
        stepStatus.classList.add("ready");
      }
    };

    reader.onerror = function () {
      showError("Could not read the selected image file. Please try another file.");
      clearPreview();
    };

    reader.readAsDataURL(file);
  }

  // ─────────────────────────────────────────────────────────────
  //  Clear / Reset Preview (Requirement 9)
  // ─────────────────────────────────────────────────────────────
  function clearPreview() {
    if (fileInput) fileInput.value = "";
    if (previewImage) previewImage.src = "#";
    if (previewName) previewName.textContent = "";
    if (previewSize) previewSize.textContent = "";
    if (previewType) previewType.textContent = "";

    if (dropContent) dropContent.style.display = "flex";
    if (previewWrapper) previewWrapper.style.display = "none";
    if (classifyBtn) classifyBtn.disabled = true;
    if (resetAllBtn) resetAllBtn.style.display = "none";

    if (stepStatus) {
      stepStatus.textContent = "No image selected";
      stepStatus.classList.remove("ready");
    }

    hideError();
  }

  // ─────────────────────────────────────────────────────────────
  //  Drop Zone & File Input Handlers (Requirement 3)
  // ─────────────────────────────────────────────────────────────
  if (dropZone && fileInput) {

    // File input change
    fileInput.addEventListener("change", function () {
      if (this.files && this.files[0]) {
        showPreview(this.files[0]);
      }
    });

    // Drag-over styling with counter to prevent child flicker
    let dragCounter = 0;

    dropZone.addEventListener("dragenter", function (e) {
      e.preventDefault();
      dragCounter++;
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", function () {
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        dropZone.classList.remove("drag-over");
      }
    });

    dropZone.addEventListener("drop", function (e) {
      e.preventDefault();
      dragCounter = 0;
      dropZone.classList.remove("drag-over");

      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        const file = dt.files[0];
        try {
          // Assign to fileInput so standard form POST carries the payload
          const container = new DataTransfer();
          container.items.add(file);
          fileInput.files = container.files;
        } catch (err) {
          // Fallback if DataTransfer not supported
          console.warn("DataTransfer assignment fallback:", err);
        }
        showPreview(file);
      }
    });

    // Clicking drop zone opens file picker unless clicking a button/label
    dropZone.addEventListener("click", function (e) {
      const tag = e.target.tagName.toLowerCase();
      if (tag === "button" || tag === "label" || e.target.closest("button") || e.target.closest("label")) {
        return;
      }
      fileInput.click();
    });

    // Keyboard accessibility for drop zone
    dropZone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });
  }

  // Remove & Reset buttons (Requirement 9)
  if (removeBtn) {
    removeBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      clearPreview();
    });
  }

  if (resetAllBtn) {
    resetAllBtn.addEventListener("click", function (e) {
      e.preventDefault();
      clearPreview();
    });
  }

  // ─────────────────────────────────────────────────────────────
  //  Model Selector Interaction (Requirement 2)
  // ─────────────────────────────────────────────────────────────
  function updateModelSelection() {
    const selectedRadio = document.querySelector('input[name="model_choice"]:checked');
    const modelChoice = selectedRadio ? selectedRadio.value : "vit";

    modelOptions.forEach(function (opt) {
      const radio = opt.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        opt.classList.add("active");
      } else {
        opt.classList.remove("active");
      }
    });

    // Update dynamic loading text
    if (loadingTitle && loadingSubtitle) {
      if (modelChoice === "vit") {
        loadingTitle.textContent = "Classifying with Vision Transformer...";
        loadingSubtitle.textContent = "Evaluating image across 102 flower categories using Self-Attention";
        if (classifyBtnText) classifyBtnText.textContent = "Classify with ViT-Base";
      } else {
        loadingTitle.textContent = "Classifying with Custom PyTorch CNN...";
        loadingSubtitle.textContent = "Evaluating image across 102 flower categories using Convolutional Net";
        if (classifyBtnText) classifyBtnText.textContent = "Classify with Custom CNN";
      }
    }
  }

  if (modelRadios.length > 0) {
    modelRadios.forEach(function (radio) {
      radio.addEventListener("change", updateModelSelection);
    });
    // Initialize active states
    updateModelSelection();
  }

  // ─────────────────────────────────────────────────────────────
  //  Form Submission & Loading State (Requirement 4)
  // ─────────────────────────────────────────────────────────────
  if (uploadForm) {
    uploadForm.addEventListener("submit", function (e) {
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        showError("Please select a flower image before clicking Classify.");
        return;
      }

      const file = fileInput.files[0];
      if (!validateFile(file)) {
        e.preventDefault();
        return;
      }

      // Display loading overlay
      if (loadingOverlay) {
        loadingOverlay.style.display = "flex";
      }
      if (classifyBtn) {
        classifyBtn.disabled = true;
      }
    });
  }

  // Handle browser Back/Forward cache (bfcache)
  window.addEventListener("pageshow", function (event) {
    if (loadingOverlay) {
      loadingOverlay.style.display = "none";
    }
    if (classifyBtn && fileInput && fileInput.files && fileInput.files.length > 0) {
      classifyBtn.disabled = false;
    }
  });

  // ─────────────────────────────────────────────────────────────
  //  Session History Manager (Requirements 8 & 9)
  //  Data is stored in sessionStorage only — never stored permanently!
  // ─────────────────────────────────────────────────────────────
  function getSessionHistory() {
    try {
      const data = sessionStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch (err) {
      console.warn("Could not read from sessionStorage:", err);
      return [];
    }
  }

  function renderSessionHistory() {
    if (!historySection || !historyGrid) return;

    const history = getSessionHistory();

    if (!history || history.length === 0) {
      historySection.style.display = "none";
      return;
    }

    historySection.style.display = "block";
    historyGrid.innerHTML = "";

    if (historyCount) {
      historyCount.textContent = `${history.length} ${history.length === 1 ? "prediction" : "predictions"}`;
    }

    history.forEach(function (item) {
      const card = document.createElement("div");
      card.className = "history-item-card";

      const modelBadgeClass = item.modelChoice === "vit" ? "vit-pill" : "cnn-pill";
      const modelIcon = item.modelChoice === "vit" ? "⚡" : "🧠";
      const modelLabel = item.modelChoice === "vit" ? "ViT-Base" : "Custom CNN";

      // Card layout
      card.innerHTML = `
        <div class="history-item-thumb-wrap">
          <img src="${item.imageUrl}" alt="${item.className}" class="history-item-thumb"
               onerror="this.parentElement.innerHTML='<span class=\\'history-thumb-fallback\\'>🌸</span>'" />
        </div>
        <div class="history-item-content">
          <div class="history-item-header">
            <h4 class="history-item-name">${item.className}</h4>
            <span class="history-item-conf">${item.confidence}%</span>
          </div>
          <div class="history-progress-mini">
            <div class="history-progress-bar" style="width: ${item.confidence}%;"></div>
          </div>
          <div class="history-item-footer">
            <span class="history-model-tag ${modelBadgeClass}">${modelIcon} ${modelLabel}</span>
            <span class="history-timestamp">${item.timestamp}</span>
          </div>
        </div>
      `;

      historyGrid.appendChild(card);
    });
  }

  // Clear Session History (Requirement 9)
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", function () {
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch (err) {
        console.warn("Failed to clear sessionStorage:", err);
      }
      if (historySection) {
        historySection.style.display = "none";
      }
    });
  }

  // Initialize session history display on page load
  window.addEventListener("DOMContentLoaded", function () {
    renderSessionHistory();
  });

})();
