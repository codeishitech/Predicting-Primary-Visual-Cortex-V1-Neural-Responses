# Predicting Primary Visual Cortex (V1) Neural Responses from Natural Images
### A Forward Encoding Model using Pretrained CNN Features and Receptive-Field-Aware Pooling

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-CRCNS%20pvc--8-green)](https://crcns.org/data-sets/vc/pvc-8)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This project builds a **forward neural encoding model** that predicts how individual neurons in the primary visual cortex (V1) of macaque monkeys respond to natural images. Given a photograph as input, the model outputs predicted firing rates for a population of real, simultaneously recorded V1 neurons.

The central finding: replacing global average pooling with **receptive-field-aware (RF-cropped) pooling** — restricting CNN feature extraction to the image region each neuron actually sees — improves mean explained variance by **~2.5x** and improves prediction quality for **24 out of 26** neurons.

This work is directly motivated by the encoding problem faced by cortical visual prosthetics such as Neuralink's Blindsight implant: to electrically stimulate V1 in a way that produces meaningful visual percepts, you first need a reliable model of how V1 responds to visual input. This project builds the forward direction of that model.

---

## Key Results

| Model | Mean Explained Variance | Best Neuron EV | Neurons Improved |
|---|---|---|---|
| Baseline (global average pooling) | 0.068 | 0.322 (Neuron 0) | — |
| RF-cropped pooling | **0.168** | **0.394** (Neuron 20) | **24 / 26** |

> Explained variance (EV) measures how much of a neuron's real response variation the model captures beyond a flat-average baseline. EV = 1.0 is perfect prediction; EV = 0 means no better than always predicting the mean.

---

## Scientific Background

### Why V1?

V1 (primary visual cortex) is the first cortical processing stage for visual information and is organized **retinotopically** — each point in the visual field maps to a specific location on the V1 cortical surface. This spatial organization is exactly why V1 is the implant target for cortical visual prostheses: electrically stimulating a given patch of V1 causes the patient to perceive a spot of light (a phosphene) at the corresponding location in their visual field, even without functioning eyes or optic nerves.

### The Forward Encoding Model

A forward encoding model answers: *given a visual stimulus, what neural response does it produce?*

This is the computational equivalent of the brain's own processing: image → neural activation. A visual prosthetic must solve the **inverse** problem: desired neural activation → what stimulation pattern produces it? A reliable forward model is the necessary first building block for that inversion.

### Why Pretrained CNN Features?

Convolutional neural networks trained on large image datasets learn hierarchical visual feature representations — edges and orientations in early layers, textures and shapes in middle layers, semantic object features in later layers — that closely resemble the representational hierarchy of the primate visual cortex. Using a pretrained CNN (VGG-16, trained on ImageNet) as a fixed feature extractor, rather than training from scratch, is the field-standard approach for this small-data regime and matches the methodology of published V1 encoding work (Uran et al., 2022; McFarland et al.).

### The Receptive Field Insight

V1 neurons have small, spatially localized **receptive fields** — they respond only to visual input within a specific patch of the visual field. Global average pooling, which averages CNN features across the entire image, mixes irrelevant content (from outside a neuron's receptive field) into the feature vector used to predict that neuron's response. This dilutes the predictive signal.

RF-cropped pooling fixes this by extracting features only from the image region corresponding to each neuron's actual receptive field location, as recorded in the dataset's RF_SPATIAL metadata. This is a direct computational analogue of how V1 neurons actually work.

This finding connects to **surround/contextual modulation** — the documented phenomenon (and the subject of the original dataset paper) that what lies *outside* a neuron's receptive field also modulates its response. The clear improvement from RF-only features demonstrates that the primary signal comes from within the RF; whether surround context adds further predictive power beyond that is left as an open extension (see Future Work).

---

## Dataset

This project uses the **CRCNS pvc-8** dataset: multi-electrode (Utah array) recordings from V1 in anesthetized macaque monkeys, while natural images and gratings were flashed on screen.

**To obtain the data:**
1. Register for a free account at [crcns.org](https://crcns.org) (approval usually same-day)
2. Navigate to: Data Sets → Visual Cortex → pvc-8
3. Download `01.mat` from the `data/` folder and place it in the project root

**Dataset details (Session 01, used in this project):**
- 102 recorded electrode channels; 26 spike-sorted, well-centered neurons (INDCENT filter)
- 956 total stimuli (540 natural images + gratings); only the 540 natural images used here
- 20 repeated trials per image; 106 ms stimulus presentation window
- Spike timing resolution: 1 millisecond

**Dataset citation (required if you use this data):**
> Adam Kohn, Ruben Coen-Cagli (2015). *Multi-electrode recordings of anesthetized macaque V1 responses to static natural images and gratings.* CRCNS.org. http://dx.doi.org/10.6080/K0SB43P8

> Please also consult Adam Kohn (adam.kohn@einstein.yu.edu) before publication, per CRCNS terms.

---

## Methods

### Pipeline Overview

```
Natural images (320x320, grayscale)
         │
         ▼
  VGG-16 (pretrained, frozen)
  Layer 16 output: [batch, 256, 28, 28]
         │
         ├──── Global average pooling ──────→ [540, 256] shared feature matrix
         │           (Baseline model)
         │
         └──── RF-cropped pooling ──────────→ [540, 256] per-neuron feature matrix
                    (Improved model)              (one matrix per neuron)
         │
         ▼
  Ridge Regression (RidgeCV, alpha=1000)
  One model per neuron
         │
         ▼
  Predicted firing rates [108 test images x 26 neurons]
         │
         ▼
  Evaluation: Explained Variance per neuron
```

### Feature Extraction

- **Backbone:** VGG-16 pretrained on ImageNet (frozen, no fine-tuning)
- **Layer tapped:** Layer 16 (end of convolutional block 3, 256 channels)
  - Chosen based on Uran et al. (2022) finding that middle VGG-16 layers best predict V1 firing rates
- **Preprocessing:** resize to 224×224, grayscale replicated to 3 channels, ImageNet normalization
- **Pooling (baseline):** global average pool across full 28×28 spatial grid → [256]
- **Pooling (RF-cropped):** average pool within spatial crop corresponding to each neuron's RF → [256]

### Receptive Field Coordinate Mapping

RF_SPATIAL positions (in degrees of visual angle) are converted to feature map cell indices using the image field-of-view (120° × 95°):

```
degrees_per_cell_W = 120° / 28 = 4.29° per cell
degrees_per_cell_H = 95°  / 28 = 3.39° per cell

cell_x = (RF_x / degrees_per_cell_W) + feat_W / 2
cell_y = (RF_y / degrees_per_cell_H) + feat_H / 2
```

### Regression

- **Model:** Ridge regression (L2 regularization)
- **Hyperparameter tuning:** RidgeCV over [1, 10, 100, 1000, 5000, 10000]; best alpha = 1000
- **Train/test split:** 80/20 by image (432 train / 108 test), stratified by random_state=42
- **One model per neuron** (each neuron has its own feature matrix and its own fitted model)

### Data Quality Finding

During analysis, RF_SPATIAL was found to contain only 15 unique receptive field positions across 26 centered neurons (11 duplicates). This is consistent with closely-spaced Utah array electrodes capturing correlated or shared underlying neural signal and is a documented artifact of dense multi-electrode recordings rather than a preprocessing error. All 26 neurons were retained for the encoding model (which uses spike data, not RF metadata); the duplicate finding was noted and accounted for in the RF-distance correlation analysis.

---

## Repository Structure

```
v1-encoding-model/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_and_response.ipynb     # Data loading, spike extraction, response matrix
│   ├── 02_baseline_model.ipynb        # VGG-16 global pooling, Ridge baseline, EV evaluation
│   ├── 03_rf_cropped_model.ipynb      # RF coordinate mapping, cropped pooling, comparison
│   ├── 04_inversion.ipynb             # Activation maximization (in progress)
│   └── 05_layer_comparison.ipynb      # Early/middle/late layer EV comparison (in progress)
│
├── results/
│   ├── figures/
│   │   └── rf_cropped_vs_baseline.png # Main comparison figure
│   └── reports/
│       ├── Session1_Report.docx        # Peer-review session log 1
│       └── Session2_Report.docx        # Peer-review session log 2
│
├── requirements.txt
└── .gitignore
```

**Note:** `.mat` data files are not included in this repository (size + redistribution terms). See Dataset section above for download instructions.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/v1-encoding-model.git
cd v1-encoding-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download 01.mat from CRCNS pvc-8 and place in project root

# 4. Run notebooks in order
# Open in Jupyter or Google Colab:
# notebooks/01_data_and_response.ipynb
# notebooks/02_baseline_model.ipynb
# notebooks/03_rf_cropped_model.ipynb
```

---

## Future Work

- **Activation maximization (inversion):** gradient ascent on input image to find each neuron's optimal stimulus — the simplified computational analogue of the visual prosthetic stimulation design problem
- **VGG-16 layer comparison:** repeat feature extraction at early (layer 4) and late (layer 23/28) layers to test whether this dataset reproduces Uran et al.'s finding that middle layers predict V1 best
- **Surround modulation test:** compare RF-only vs. full-image features using the two image sizes in this dataset (1° and 3-6.7°) to directly test whether surround context adds predictive power beyond the RF — the original dataset paper's research question, applied to this modeling approach
- **Multi-session replication:** run the full pipeline on sessions 02-04 (Animal 2) to test generalization across independent recording sessions
- **Receptive field size analysis:** test whether RF size (var_x, var_y from RF_SPATIAL) predicts model performance, independent of position

---

## References

1. **Coen-Cagli R, Kohn A, Schwartz O** (2015). Flexible Gating of Contextual Influences in Natural Vision. *Nature Neuroscience.* doi:10.1038/nn.4128 *(source of the dataset used in this project)*

2. **Uran C, et al.** (2022). Predictive Coding of Natural Images by V1 Firing Rates and Rhythmic Synchronization. *Neuron.* *(closest methodological precedent for VGG-16 features → V1 firing rate prediction)*

3. **McFarland JM, Cui Y, Butts DA.** Using Deep Learning to Reveal the Neural Code for Images in Primary Visual Cortex. arXiv:1706.06208 *(closest precedent for the overall project goal)*

4. **Simonyan K, Zisserman A** (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556 *(VGG-16 architecture)*

5. **Hubel DH, Wiesel TN** (1968). Receptive fields and functional architecture of monkey striate cortex. *Journal of Physiology.* *(foundational V1 receptive field work)*

---

## License

MIT License. See LICENSE for details.

Note: The CRCNS pvc-8 dataset has its own usage terms — please read them at crcns.org and cite appropriately if you use this work.

---

*This project was developed as part of an independent study in computational neuroscience and brain-computer interface research.*
