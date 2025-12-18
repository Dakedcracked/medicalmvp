# 🧠 Neuron AI: Technical Documentation & Resource Index

## 1. Core System Scripts
These are the production-ready scripts that power the Neuron AI engine.

| Script | Purpose | Location |
| :--- | :--- | :--- |
| **API Server** | The main backend brain. Handles inference, TTA, and CLAHE preprocessing. | [`api_server.py`](../api_server.py) |
| **Universal Loader** | Automatically downloads SOTA weights or trains models if data exists. | [`training_scripts/train_model.py`](../training_scripts/train_model.py) |

---

## 2. SOTA Training Recipes (Source Code)
These scripts contain the exact methodologies used by researchers to achieve >95% accuracy. Use these to reproduce results from scratch.

### A. Chest X-Ray (DenseNet121)
*   **Script:** [`training_scripts/train_xray_sota.py`](../training_scripts/train_xray_sota.py)
*   **Methodology:** CheXpert Standard (BCE Loss, DenseNet121-All).
*   **Key Fixes:** Uses `torchxrayvision` normalization to prevent "Artifact Bias".

### B. Skin Cancer (EfficientNet-B4)
*   **Script:** [`training_scripts/train_skin_sota.py`](../training_scripts/train_skin_sota.py)
*   **Methodology:** ISIC Winning Solution (CutMix, Class Balancing).
*   **Key Fixes:** "Microscope Simulation" augmentation to handle hair/ruler artifacts.

### C. Brain MRI (ResNet50)
*   **Script:** [`training_scripts/train_mri_sota.py`](../training_scripts/train_mri_sota.py)
*   **Methodology:** Slice Classification (ResNet50).
*   **Key Fixes:** Intensity Normalization to handle different MRI scanner contrasts.

---

## 3. Model Weights (The "Brains")
Once you run `python training_scripts/train_model.py`, the high-accuracy weights will be saved here:

| Modality | Model File | Description |
| :--- | :--- | :--- |
| **X-Ray** | `weights/xray/densenet_sota.pth` | The 121-layer "Super-Model" trained on 300k+ images. |
| **Skin** | `weights/skin/efficientnet_skin.pth` | Optimized for texture/lesion analysis. |
| **MRI** | `weights/mri/resnet50_mri.pth` | Optimized for grayscale brain structure. |

---

## 4. Vulnerability Fixes Implemented

### A. "Artifact Bias" (The Tube Problem)
*   **Fix:** Implemented **CLAHE (Contrast Limited Adaptive Histogram Equalization)** in `api_server.py`.
*   **Why:** It forces the model to look at local tissue texture instead of global brightness (which often correlates with "sick" patients having portable scans).

### B. "2D Shadow" (The Overlap Problem)
*   **Fix:** Implemented **Test-Time Augmentation (TTA)**.
*   **Why:** By analyzing the image at 3 different zooms/flips, we reduce the chance that a rib overlap is mistaken for a tumor.

### C. "Black Box" Liability
*   **Fix:** **Grad-CAM** is mandatory.
*   **Why:** Every prediction now comes with a heatmap. If the heatmap is wrong, the doctor knows to reject the diagnosis immediately.
