# Explainable AI for Diabetic Retinopathy Screening

An explainable deep-learning system for **diabetic retinopathy (DR) severity classification from retinal fundus images**, with **Grad-CAM visual explanations** to show the retinal regions that most influenced the model's prediction.

This project was developed as a **hackathon/research prototype** with the goal of supporting faster preliminary screening and helping healthcare workers understand why an AI model produced a particular prediction.

> **Disclaimer:** This is a research/hackathon prototype and is **not a clinically validated medical diagnostic system**. Model predictions should not replace examination or diagnosis by a qualified ophthalmologist.

---

## Overview

Diabetic retinopathy is a complication of diabetes that can progressively damage the retina. Early stages may not produce obvious symptoms, making regular retinal screening important.

In many screening settings, the number of retinal images can be large compared with the availability of trained specialists. This project explores an AI-assisted screening pipeline that can:

1. Analyze a retinal fundus image.
2. Classify diabetic retinopathy severity into five grades.
3. Estimate the prediction confidence.
4. Categorize the result into a project-level **referable / non-referable** screening group.
5. Generate a **Grad-CAM heatmap** showing regions that contributed most strongly to the model's prediction.

The goal is not to replace ophthalmologists, but to explore how explainable AI could support preliminary screening workflows.

---

## Key Features

* **5-class diabetic retinopathy classification**
* **EfficientNet-B0** image classification model
* **ImageNet pretrained weights** with transfer learning
* Training using **APTOS 2019 + DDR** retinal image datasets
* **Class-weighted cross-entropy** to address class imbalance
* **Quadratic Weighted Kappa (QWK)** evaluation for ordinal severity grading
* **Grad-CAM** visual explanations
* Project-level **referable / non-referable** screening categorization
* Trained PyTorch model checkpoint included in the repository

---

## Diabetic Retinopathy Classes

The model predicts five severity levels:

| Class | Diagnosis        |
| ----: | ---------------- |
|     0 | No DR            |
|     1 | Mild DR          |
|     2 | Moderate DR      |
|     3 | Severe DR        |
|     4 | Proliferative DR |

The five-grade structure follows the labeling used by the APTOS 2019 diabetic retinopathy dataset.

---

## System Pipeline

```text
              Retinal Fundus Image
                       │
                       ▼
              Image Preprocessing
                 Resize → 224×224
                       │
                       ▼
                EfficientNet-B0
                       │
                       ▼
              5-Class Classification
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   DR Severity Prediction       Grad-CAM
          │                         │
          ▼                         ▼
   Confidence Score          Visual Explanation
          │
          ▼
 Referable / Non-Referable
       Screening
```

---

## Model Architecture

The project uses **EfficientNet-B0** with ImageNet pretrained weights.

The original classification head was replaced with a five-class linear layer:

```text
EfficientNet-B0
      │
      ├── Feature Extraction
      │
      └── Classification Head
              │
              └── 5 outputs
                   ├── No DR
                   ├── Mild DR
                   ├── Moderate DR
                   ├── Severe DR
                   └── Proliferative DR
```

### Input

```text
RGB retinal fundus image
        ↓
Resize to 224 × 224
        ↓
ImageNet normalization
```

### Data augmentation

Training images use:

* Random horizontal flip
* Random rotation
* Resize to `224 × 224`
* ImageNet normalization

Validation images are resized and normalized without random augmentation.

---

## Datasets

### APTOS 2019

The project uses the APTOS 2019 Blindness Detection dataset.

The original APTOS dataset contains retinal fundus images labeled from 0 to 4 according to diabetic retinopathy severity.

For this project, the Kaggle copy used during training contained:

```text
2,930 training images
```

with the following distribution:

| Class | Diagnosis        | Images |
| ----: | ---------------- | -----: |
|     0 | No DR            |   1434 |
|     1 | Mild DR          |    300 |
|     2 | Moderate DR      |    808 |
|     3 | Severe DR        |    154 |
|     4 | Proliferative DR |    234 |

Dataset source:

[APTOS 2019 Blindness Detection on Kaggle](https://www.kaggle.com/competitions/APTOS2019-blindness-detection/data?utm_source=chatgpt.com)

---

### DDR Dataset

The project also uses the **DDR (Diabetic Retinopathy Dataset)**.

The Kaggle copy used in this project contains:

```text
12,522 labeled retinal images
```

with the five DR severity classes.

The images were combined with the APTOS training data to improve the diversity of the training set.

Dataset source:

[DDR Dataset on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset?utm_source=chatgpt.com)

> The datasets themselves are **not included in this repository** because of their size and dataset-specific distribution/usage terms.

---

## Combined Training Dataset

The combined training set used in this project consisted of:

```text
APTOS training images
        +
DDR training images
        ↓
Combined training dataset
```

The resulting class distribution was:

| Class | Diagnosis        | Images |
| ----: | ---------------- | -----: |
|     0 | No DR            |   6760 |
|     1 | Mild DR          |    835 |
|     2 | Moderate DR      |   4613 |
|     3 | Severe DR        |    355 |
|     4 | Proliferative DR |   1010 |

Because the classes are highly imbalanced, class-weighted loss was used during training.

---

## Training

### Framework

* Python
* PyTorch
* Torchvision
* Scikit-learn
* PIL
* Matplotlib

### Optimizer

The model was trained using AdamW:

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
```

### Loss

Weighted cross-entropy loss was used to compensate for class imbalance.

The approximate class weights used for the combined dataset were:

```text
[0.4016, 3.2510, 0.5885, 7.6468, 2.6877]
```

### Image size

```text
224 × 224
```

### Batch size

```text
32
```

### Epochs

```text
5
```

---

## Results

The model was trained for five epochs.

| Epoch | Train Loss | Train Accuracy |  Train QWK | Validation Loss | Validation Accuracy | Validation QWK |
| ----: | ---------: | -------------: | ---------: | --------------: | ------------------: | -------------: |
|     1 |     1.1239 |         0.5563 |     0.6112 |          1.0795 |              0.7049 |         0.8116 |
|     2 |     0.8296 |         0.6782 |     0.7667 |          0.9719 |              0.6940 |         0.8360 |
|     3 |     0.7024 |         0.7194 |     0.8024 |          1.3317 |              0.6940 |         0.8182 |
| **4** | **0.6374** |     **0.7359** | **0.8243** |      **0.8930** |          **0.7814** |     **0.8880** |
|     5 |     0.5530 |         0.7689 |     0.8426 |          0.8297 |              0.8005 |         0.8863 |

### Best validation result

The best validation **Quadratic Weighted Kappa** was:

```text
QWK = 0.8880
```

at **epoch 4**.

The corresponding checkpoint is included in:

```text
models/
└── efficientnet_b0_aptos_ddr_best.pth
```

---

## Why Quadratic Weighted Kappa?

Diabetic retinopathy severity is an **ordinal classification problem**.

The classes have an inherent order:

```text
No DR
  ↓
Mild
  ↓
Moderate
  ↓
Severe
  ↓
Proliferative
```

Therefore, predicting class 4 when the actual class is 3 is less severe than predicting class 0 when the actual class is 4.

Quadratic Weighted Kappa (QWK) accounts for the degree of disagreement between the predicted and true severity grades and is commonly used for ordinal DR grading tasks.

---

## Explainability with Grad-CAM

A major component of this project is **explainability**.

Instead of displaying only:

```text
Prediction: Moderate DR
Confidence: 87%
```

the system also generates a Grad-CAM heatmap.

### Grad-CAM pipeline

```text
Fundus Image
     │
     ▼
EfficientNet-B0
     │
     ▼
Predicted DR Class
     │
     ▼
Gradients from Target Class
     │
     ▼
Feature Activations
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Overlay on Fundus Image
```

Grad-CAM highlights image regions that had greater influence on the model's prediction.

### Important interpretation

The Grad-CAM visualization should **not** be interpreted as proof that a specific lesion caused the diagnosis.

It is an **attribution visualization** indicating which image regions most influenced the model's output.

This distinction is important when applying explainable AI to medical imaging.

---

## Referable DR Screening

For the prototype, the five severity classes are additionally grouped into two screening categories:

```text
Class 0 → Non-referable
Class 1 → Non-referable

Class 2 → Referable
Class 3 → Referable
Class 4 → Referable
```

Therefore:

```text
0, 1 → Non-referable DR
2, 3, 4 → Referable DR
```

This is a **project-level screening rule**, not a claim of clinical diagnostic equivalence.

The purpose is to demonstrate how a multiclass severity model could provide an additional screening-oriented output.

---

## Example Output

The intended inference output contains:

```text
Diagnosis: Moderate DR
Confidence: XX.X%
Screening Status: Referable DR
```

along with:

```text
Original Fundus Image
        +
Grad-CAM Heatmap
        +
Grad-CAM Overlay
```

This provides both a model prediction and a visual explanation.

---

## Repository Structure

```text
explainable-ai-diabetic-retinopathy/
│
├── models/
│   └── efficientnet_b0_aptos_ddr_best.pth
│
├── notebooks/
│   └── SIH_model.ipynb
│
├── .gitignore
│
└── README.md
```

### `models/`

Contains the trained PyTorch model checkpoint.

```text
efficientnet_b0_aptos_ddr_best.pth
```

The checkpoint contains the trained model parameters.

### `notebooks/`

Contains the complete development/training notebook used for:

* Dataset loading
* Preprocessing
* Model creation
* Training
* Validation
* Evaluation
* Prediction
* Grad-CAM generation

---

## Running the Notebook

The notebook was developed in **Kaggle** using GPU acceleration.

The original training environment uses Kaggle dataset paths such as:

```text
/kaggle/input/...
```

Therefore, the notebook is currently intended to be reproduced in a Kaggle environment after adding the required datasets.

### Required datasets

Add:

* APTOS 2019 Blindness Detection
* DDR Dataset

Then update the dataset paths in the notebook if necessary.

---

## Requirements

The main Python libraries used by the project are:

```text
torch
torchvision
scikit-learn
pandas
numpy
Pillow
matplotlib
```

A future version of this repository will include a complete `requirements.txt` and a standalone inference application.

---

## Limitations

This project has several important limitations.

### 1. Dataset size and composition

The project uses specific Kaggle copies of the APTOS and DDR datasets rather than a newly collected clinical dataset.

The APTOS copy used for training contains 2,930 images, while the original APTOS competition dataset contains 3,662 labeled training images.

### 2. Class imbalance

Some DR severity classes contain substantially fewer images than others.

Class-weighted loss was used to reduce the effect of this imbalance, but it does not completely remove the problem.

### 3. Domain shift

Fundus cameras, acquisition settings, patient populations, image quality and clinical environments can vary considerably.

Performance on the training/validation datasets does not guarantee the same performance on images from another hospital or camera.

### 4. Explainability limitations

Grad-CAM provides an attribution map, not a clinical explanation or lesion detector.

A highlighted region should therefore not automatically be interpreted as a clinically confirmed lesion.

### 5. Clinical validation

The model has **not undergone prospective clinical validation** and should not be used for autonomous diagnosis or treatment decisions.

---

## Future Work

Planned improvements include:

* [ ] Standalone web-based inference interface
* [ ] Automatic Grad-CAM visualization during inference
* [ ] Better retinal image preprocessing
* [ ] More extensive hyperparameter tuning
* [ ] Cross-dataset evaluation
* [ ] Calibration of confidence scores
* [ ] Sensitivity/specificity evaluation for referable DR
* [ ] ROC-AUC evaluation for screening
* [ ] External clinical validation
* [ ] Model quantization
* [ ] Lightweight deployment for edge/mobile devices
* [ ] Integration with low-cost fundus-camera hardware
* [ ] Evaluation on additional retinal datasets
* [ ] More robust explainability methods

---

## Potential Deployment Concept

The long-term goal is to explore a lightweight screening system that could operate in environments where specialist availability is limited.

A possible workflow is:

```text
Low-cost Fundus Camera
          │
          ▼
   Retinal Image Capture
          │
          ▼
    AI Screening Model
          │
     ┌────┴────┐
     ▼         ▼
Prediction   Grad-CAM
     │         │
     └────┬────┘
          ▼
 Screening Result
          │
          ▼
 Ophthalmologist Review
```

The AI system would function as a **screening assistance tool**, with specialist review remaining part of the workflow.

---

## Project Objective

The core objective of this project is to demonstrate that a deep-learning model can combine:

```text
Automated Classification
          +
Ordinal Severity Prediction
          +
Explainability
          +
Screening Categorization
```

into a single retinal-image analysis pipeline.

The emphasis is on building an AI system that is not only capable of producing a prediction, but also provides a visual indication of **where the model's attention was concentrated**.

---

## Model

**Architecture:** EfficientNet-B0
**Pretraining:** ImageNet
**Input:** 224 × 224 RGB fundus image
**Output:** 5 DR severity classes
**Framework:** PyTorch
**Explainability:** Grad-CAM
**Primary ordinal metric:** Quadratic Weighted Kappa
**Best validation QWK:** **0.8880**

---

## Acknowledgements

This project makes use of publicly available diabetic retinopathy datasets, including:

* APTOS 2019 Blindness Detection
* DDR Dataset

The project is intended for educational, research and hackathon purposes.

---

## Disclaimer

This repository contains an experimental machine-learning prototype.

**It is not a medical device and has not been clinically validated.**

Predictions generated by this model should not be used as a substitute for professional medical examination, diagnosis, or treatment. Any real-world deployment would require appropriate clinical validation, regulatory review, safety assessment, and evaluation across representative patient populations and imaging environments.

---

## Author

GitHub: [@ankushmbadiger](https://github.com/ankushmbadiger)

Project repository:

https://github.com/ankushmbadiger/explainable-ai-diabetic-retinopathy
