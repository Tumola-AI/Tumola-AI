# TOUMÖLA
### Innovative System for Protection, Conservation and Valorization of Medicinal Plants using Artificial Intelligence

---

## Table of Contents

- [About the Project](#about-the-project)
- [Results](#results)
- [System Architecture](#system-architecture)
- [Datasets Used](#datasets-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Complete Pipeline](#complete-pipeline)
- [Hyperparameters](#hyperparameters)
- [Training on Kaggle](#training-on-kaggle)
- [Detailed Results](#detailed-results)


---

## Project Structure
  ├── input/                           #
│   ├── toumola_dataset_final.csv       #
│   └── images/ 


## About the Project

Guinea has exceptional plant biodiversity with more than **3,500 species**, of which nearly **1,200 have medicinal properties**. However, modernization, deforestation and lack of oral transmission are endangering these resources.

**TOUMÖLA** aims to address this dual challenge:
- Preserve an **intangible heritage** that is disappearing
- Improve **access to healthcare** through automatic identification of medicinal plants

### Objectives

| Objective | Target |
|---|---|
| Digital identification of Guinean plants | 70% of species |
| Local treatment of endemic diseases | 60% of diseases |
| Citizen awareness through digital campaigns | 80% of population |

---

## Results

| Metric | Value |
|---|---|
| **Overall Accuracy** | **95.76%** |
| Precision (macro avg) | 96% |
| Recall (macro avg) | 96% |
| F1-Score (macro avg) | 96% |
| Species covered | 55 |
| Test images | 778 |

### Performance by Species (Top 10)

| Species | Precision | Recall | F1-Score |
|---|---|---|---|
| Aloe vera | 1.00 | 1.00 | **1.00** |
| Moringa oleifera | 1.00 | 1.00 | **1.00** |
| Azadirachta indica | 0.95 | 1.00 | **0.98** |
| Mangifera indica | 0.95 | 1.00 | **0.98** |
| Citrus × limon | 0.95 | 1.00 | **0.98** |
| Jasminum officinale | 0.96 | 0.96 | **0.96** |
| Mentha spicata | 0.96 | 0.96 | **0.96** |
| Phyllanthus emblica | 1.00 | 0.93 | **0.96** |
| Oxalis corniculata | 0.94 | 1.00 | **0.97** |
| Bacopa | 0.88 | 1.00 | **0.94** |


---

## Datasets Used

### 1. Medicinal Leaf Dataset (MedLeaves)
- **Source:** Mendeley Data — Roopashree S. & Anitha J. (2020)
- **Content:** ~1500 images, 30-40 Indian medicinal species
- **Format:** JPG, white background, segmented leaves
- **DOI:** `10.17632/nnytj2v3n5.1`

### 2. Indian Medicinal Leaves Image Datasets
- **Source:** Mendeley Data — Pushpa B.R. & Shobha Rani (2023)
- **Content:** ~6900 images, 80 species, real field conditions
- **Format:** JPG, varied backgrounds, multi-resolutions
- **DOI:** `10.17632/748f8jkphb.3`

### 3. Dr. Duke's Phytochemical & Ethnobotanical Database
- **Source:** USDA — James A. Duke (1992–2016)
- **Content:** Medicinal uses, phytochemical compounds, biological activity
- **Format:** Downloadable CSV
- **License:** Creative Commons CC0 (public domain)

### 4. POWO — Plants of the World Online
- **Source:** Royal Botanic Gardens, Kew (London)
- **Content:** Botanical family, accepted scientific name, synonyms
- **Format:** REST API + CSV
- **URL:** `powo.science.kew.org`

### Final Dataset Statistics

| Statistic | Value |
|---|---|
| Total images | **7,780** |
| Unique species | **55** (after merging and cleaning) |
| Merged sources | 5 |
| Train split | 75% (~5,835 images) |
| Validation split | 15% (~1,167 images) |
| Test split | 10% (778 images) |

---

## Installation

### Prerequisites

```bash
Python >= 3.10
CUDA >= 11.8 (recommended for GPU training)
```

### Installing Dependencies

```bash
git clone https://github.com/your-username/toumola.git
cd toumola

pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
plotly>=5.15.0
requests>=2.31.0
tqdm>=4.65.0
albumentations>=1.3.0
torchmetrics>=1.0.0
```
---

## Complete Pipeline

### Step 1 — Dataset Construction

### Step 2 — Training (Kaggle)

See section [Training on Kaggle](#training-on-kaggle).

### Step 3 — Local Inference

```bash
python 7_inference_gradio.py \
    --model  toumola_model/toumola_final.pth \
    --csv    output/toumola_enrichi.csv \
    --labels toumola_model/labels.json
```

Open `http://localhost:7860` in your browser.

---

## Hyperparameters

### Architecture

| Parameter | Value |
|---|---|
| Model | EfficientNetB3 |
| Pre-training | ImageNet |
| Input size | 224 × 224 px |
| Number of classes | 55 |

### Training

| Parameter | Value | Description |
|---|---|---|
| Epochs | 30 | Total number of passes |
| Batch size | 32 | Images per batch |
| Optimizer | AdamW | With weight decay |
| Learning rate Phase 1 | 1e-4 | Frozen backbone |
| Learning rate Phase 2 | 1e-5 | Unfrozen backbone (epoch 10+) |
| LR minimum | 1e-6 | Scheduler floor |
| Weight decay | 1e-4 | L2 regularization |
| Scheduler | CosineAnnealingLR | Cosine decay |
| Loss function | CrossEntropyLoss | Multi-class |
| Label smoothing | 0.1 | Anti over-confidence |
| Early stopping | patience=7 | Automatic stop |
| Unfreeze epoch | 10 | Phase 1 → Phase 2 |

### Two-Phase Fine-tuning Strategy

```
Phase 1 (Epochs 1-10): FROZEN Backbone
  → Only classifier is trained
  → Fast learning of base features
  → LR = 1e-4

Phase 2 (Epochs 10-30): UNFROZEN Backbone
  → Complete network fine-tuning
  → Adaptation to medicinal leaves
  → LR = 1e-5 (reduced ×10)
```

### Data Augmentation

| Transformation | Value |
|---|---|
| Resize + RandomCrop | 256 → 224 px |
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.2 |
| RandomRotation | ±30° |
| ColorJitter | brightness/contrast/saturation = 0.3 |
| RandomGrayscale | p = 0.05 |
| Normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

---

## Training on Kaggle

### Prerequisites
1. Free Kaggle account
2. GPU enabled: **Settings → Accelerator → GPU T4 x2**

### Steps

```
1. Upload to Kaggle:
   - toumola_dataset_final.csv
   - images.zip (output/images/ folder zipped)

2. Import notebook:
   kaggle/toumola.ipynb

3. Enable GPU in Settings

4. Run All (~45 minutes)

5. Download from Output:
   - toumola_final.pth
   - labels.json
   - plant_info.json
   - training_curves.png
   - confusion_matrix.png
   - classification_report.txt
```

---

## Detailed Results

### Training Curves

The curves show clearly visible two-phase training:
- **Phase 1 (epochs 1-10)**: rapid rise from 13% → 87% accuracy
- **Phase 2 (epochs 10-30)**: progressive fine-tuning 87% → 95.76%
- **No overfitting**: train ≈ validation throughout

### Complete Classification Report

```
Accuracy: 95.76%

                    precision  recall  f1-score  support
       Aloe vera     1.00      1.00     1.00       16
  Moringa oleifera   1.00      1.00     1.00        8
   Lawsonia inermis  0.75      0.60     0.67       15  ← difficult case
         ...
       macro avg     0.96      0.96     0.96      778
    weighted avg     0.96      0.96     0.96      778
```

> **Note:** macro avg ≈ weighted avg confirms that the test dataset is well balanced.

---

## Medical Disclaimer

> Information provided by TOUMÖLA is **for educational purposes only**.
> Always consult a healthcare professional before using medicinal plants for therapeutic purposes.
> TOUMÖLA does not replace medical advice.

---

## References

```bibtex
@dataset{roopashree2020medicinal,
  title     = {Medicinal Leaf Dataset},
  author    = {Roopashree, S. and Anitha, J.},
  year      = {2020},
  publisher = {Mendeley Data},
  doi       = {10.17632/nnytj2v3n5.1}
}

@dataset{pushpa2023indian,
  title     = {Indian Medicinal Leaves Image Datasets},
  author    = {Pushpa, B.R. and Shobha Rani},
  year      = {2023},
  publisher = {Mendeley Data},
  doi       = {10.17632/748f8jkphb.3}
}

@database{duke1992phytochemical,
  title     = {Dr. Duke's Phytochemical and Ethnobotanical Databases},
  author    = {Duke, James A.},
  year      = {1992--2016},
  publisher = {USDA Agricultural Research Service},
  url       = {https://phytochem.nal.usda.gov/}
}

@database{powo2024,
  title     = {Plants of the World Online},
  author    = {{Royal Botanic Gardens, Kew}},
  year      = {2024},
  url       = {https://powo.science.kew.org/}
}

@software{ultralytics2024yolo,
  title  = {YOLOv12 Documentation},
  author = {Ultralytics},
  url    = {https://docs.ultralytics.com}
}
```

# Tumola-AI
Toumola, the Kpèlè term for "medicinal plants", is an open-source hybrid architecture integrating a fine-tuned EfficientNet-B3 computer vision module for taxonomic identification with a grounded Retrieval-Augmented Generation (RAG) system.
