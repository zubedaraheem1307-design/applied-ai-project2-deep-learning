# Project 2: Deep Learning Applications

**Name:** Zubeda Raheem
**Subject:** Introduction to Applied AI

## Project 2, Week 5: Scene Classification with Transfer Learning

- **Dataset:** Intel Image Classification, 6 classes. The training folder has 14,034 images, split 85% / 15% (seed 42) into 11,929 train and 2,105 validation images. The separate test folder has 3,000 images.
- **Model:** ResNet-18 pretrained on ImageNet, with a new 6-class head. Images resized to 224 x 224 and normalised with ImageNet mean and standard deviation.
- **Feature extraction (3,078 trainable parameters):** val accuracy 0.901
- **Fine-tuning layer4 (8,396,806 trainable parameters, about 8.4M):** val accuracy 0.931
- **Test accuracy (used once):** 0.935 (2,805 of 3,000 correct); weakest class: glacier (recall 0.870), confused with mountain
- **Most instructive error:** A railway hall with tracks, labelled "buildings", was predicted as "street" with only 0.53 confidence, which shows that some labels fit neither class well.
- **Next:** my own domain in Week 6: chest X-ray classification (Normal vs Pneumonia), see the proposal below

### Method and Decisions
- **Augmentation on training data only:** random resized crop (scale 0.7 to 1.0), horizontal flip, and brightness/contrast jitter of 0.2. Validation and test images were only resized and normalised.
- **Validation set kept separate from test:** the validation set chose the best epoch and compared feature extraction with fine-tuning, so the test set played no part in any choice.
- **Feature extraction first:** the whole ResNet-18 body was frozen and only the new head was trained (512 x 6 weights + 6 biases = 3,078 parameters, about 0.03% of the 11,179,590 total) for 3 epochs with Adam at a learning rate of 1e-3.
- **Then fine-tuning:** layer4 was unfrozen and trained with a smaller learning rate (1e-4) than the new head (1e-3), so the pretrained weights were adjusted, not destroyed. This ran for 3 more epochs.
- **Best epoch kept:** the weights from the best validation epoch were restored after each phase. For fine-tuning, that was epoch 5 (val accuracy 0.931).
- **Test run once:** the test set was evaluated a single time after all choices were made.

### Results
| Phase | Epoch | Train acc | Val acc | Val loss |
|---|---|---|---|---|
| Feature extraction | 1 | 0.809 | 0.894 | 0.327 |
| Feature extraction | 2 | 0.890 | 0.899 | 0.290 |
| Feature extraction | 3 | 0.895 | 0.901 | 0.276 |
| Fine-tuning | 4 | 0.905 | 0.926 | 0.227 |
| Fine-tuning | 5 | 0.938 | 0.931 | 0.204 |
| Fine-tuning | 6 | 0.955 | 0.926 | 0.218 |

- Fine-tuning improved best validation accuracy from 0.901 to 0.931 (+3.0 percentage points).
- In epoch 6 the validation loss rose (0.204 to 0.218) while the training loss kept falling, a mild sign of overfitting. Keeping the best epoch (5) avoided it.
- Training accuracy was below validation accuracy in epochs 1 to 4 because training images are augmented and validation images are clean.

**Per-class test results**

| Class | Train images | Test images | Recall |
|---|---|---|---|
| buildings | 2,191 | 437 | 0.899 |
| forest | 2,271 | 474 | 0.987 |
| glacier | 2,404 | 553 | 0.870 |
| mountain | 2,512 | 525 | 0.918 |
| sea | 2,274 | 510 | 0.980 |
| street | 2,382 | 501 | 0.960 |

### Error Analysis
- **Most confused pair:** glacier and mountain. 56 glacier images were predicted as mountain and 31 mountain images as glacier (87 errors). This confirms my Part 1 prediction.
- **Second most confused pair:** buildings and street (38 + 17 = 55 errors). Many of these images show a building together with a road or cars, so some labels are debatable.
- **Error gallery:** the first 8 errors in test order are all true-class buildings. They include a stone house by a stream predicted as mountain (0.70 confidence) and the railway hall mentioned above.

## Week 6 Proposal
**Problem:** Pneumonia is a serious lung infection in children, and diagnosing it from chest X-rays depends on scarce expert radiologists. For Project 2, I will build an image classifier that labels a chest X-ray as Normal or Pneumonia. I will use ResNet-18 transfer learning with the same pipeline as Week 5: augmentation on training data only, feature extraction, then fine-tuning, and a single final test. Because pneumonia images outnumber normal ones about 3 to 1, I will report per-class recall and the confusion matrix, not just accuracy. **Who benefits:** clinics with few radiologists, as a second-opinion screening aid. This is an educational project, not a diagnostic tool. **Dataset:** Chest X-Ray Images (Pneumonia) by Paul Mooney on Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), 5,863 JPEG images in 2 classes (Normal and Pneumonia), licensed CC BY 4.0. Source: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell, 2018.

> **Update (Week 6):** I changed the project dataset. Instead of the shared chest X-ray reference dataset, I chose my own domain, as the lab encourages. The final project is malaria parasite detection in single blood-cell images, using the Cell Images for Detecting Malaria dataset (US National Institutes of Health, hosted on Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria): 27,558 images in 2 balanced classes (Parasitized and Uninfected, 13,779 each), from 200 patients. The method is unchanged: ResNet-18 transfer learning, first training only the head and then fine-tuning the last block, with the best epoch chosen by macro-F1. Because the classes are balanced, the main risk is not imbalance but patient leakage, so I split the data by patient and report macro-F1, sensitivity, specificity and PPV. The final results, data audit and Grad-CAM check are in the Week 6 section further down this README. It is still an educational project and not a diagnostic tool.


## File
- **Notebook in this repo:** [Week5_Transfer_Learning.ipynb](week-5-transfer-learning.ipynb): full notebook with outputs and written answers

# Week 6: Malaria parasite detection in blood-cell images with ResNet-18

A ResNet-18 pretrained on ImageNet and fine-tuned to classify single red blood cells as **Parasitized** or **Uninfected**, with a patient-level split, a threshold set for 95% recall, and a Grad-CAM check for shortcuts.

**Headline result (test set, used once):** macro-F1 0.969, sensitivity 0.982 and specificity 0.942 at threshold 0.154.

---

## 1. Datasheet

### Who benefits and why it matters
**Who benefits:** Microscopists and community health workers who screen thin blood smears for malaria, especially in clinics with few trained microscopists.

**Why it matters:** WHO estimates 282 million malaria cases and 610,000 deaths in 2024, about 95% of the deaths in Africa and most of them children under 5. A missed infection (a false negative) delays treatment, so the cost of an error falls on the patient.

### Dataset
| Item | Value |
|---|---|
| Name | Cell Images for Detecting Malaria |
| Source | US National Institutes of Health (NIH), hosted on Kaggle |
| Link | https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria |
| Licence | Not stated on the Kaggle page ("Unknown"); data originates from the US National Institutes of Health (NIH), see the source link |
| Images | 27,558 |
| Classes | 2: Parasitized 13,779 and Uninfected 13,779 |
| Labels by | The dataset's authors (NIH, Rajaraman et al., 2018) |
| Image type | All RGB (300 sampled); width 61 to 229 px (median 130), height 82 to 187 px (median 133) |
| Patients | 200 (150 infected and 50 uninfected, as described by the source) |

The images are single red blood cells cut out of Giemsa-stained thin smears. Each file name starts with a C-number (for example `C100P61ThinF_IMG_...` gives C100), which I extracted with the pattern `^(C\d+)` and treated as the patient id for the split. All 27,558 files matched (0 unmatched), which gives 200 groups and 137.79 images per group on average (the largest, C68, has 702 images). In my audit 150 groups contribute both classes and 50 contribute only Uninfected cells, which matches the source's description of 150 infected and 50 uninfected patients.

### First shortcut suspect
Parasitized cells show dark purple spots, which is the real signal. The stain colour (pink to blue-grey) and the black padding around each cell vary from slide to slide and do not depend on the parasite. Because 50 patients contribute only Uninfected cells, a model could learn those patients' slide look and link it to "Uninfected". I tested this with Grad-CAM (Section 2.4).

### Data audit
| Check | Result |
|---|---|
| Class balance | Imbalance ratio 1.0 (13,779 / 13,779) |
| Exact duplicates | 0 duplicate sets, 0 extra copies (file size first, then MD5 on the 23,699 files that share a size with another file) |
| Near-duplicates (32 x 32 standardised fingerprint, against train) | At threshold 0.10: 234 of 3,335 test images and 249 of 5,105 validation images flagged. At 0.05: 0 in both. Closest distance 0.064 to 0.072 (recompressed copies score about 0.007) |
| Inspection of flagged pairs | The closest pairs are different pale, round cells, so the 0.10 flags are look-alikes, not copies. Nothing was removed |
| Split | By patient (StratifiedGroupKFold), 0 shared patients between train and validation, train and test, and validation and test |

**Answer in one line:** exact duplicates 0; test images with a near-duplicate in train 0 at the 0.05 threshold (234 were flagged at the looser 0.10 threshold, but the closest pairs I inspected were different look-alike cells, not copies, so nothing was removed).

| Split | Images | Patients | Parasitized | Uninfected |
|---|---|---|---|---|
| Train | 19,118 | 139 | 9,586 | 9,532 |
| Validation | 5,105 | 33 | 2,853 | 2,252 |
| Test | 3,335 | 28 | 1,340 | 1,995 |

The test set is 40.2% parasitized, because the grouped split could not make it exactly 50/50.

**Random split (what would have happened):** with about 138 cells per patient, a random split would put cells from the same patient and slide on both sides, so the model could recognise a slide's stain and background instead of the parasite, and the test score would be higher than on new patients.

### Imbalance handling
Weighted loss (`USE_SAMPLER = False`) with weight = N / (K x n_c), N = 19,118 and K = 2:
- Parasitized: 19,118 / (2 x 9,586) = 19,118 / 19,172 = **0.997**
- Uninfected: 19,118 / (2 x 9,532) = 19,118 / 19,064 = **1.003**

The training set is almost balanced, so the loss is effectively unweighted. I keep the weights because they cost nothing and would matter if the split were uneven.

**Augmentation:** random resized crop (scale 0.8 to 1.0), rotation up to 10 degrees, brightness and contrast jitter of 0.15, and horizontal and vertical flips (added to the lab's list, because a cell has no left-right or up-down orientation). I did not add hue or saturation jitter, because the purple stain of the parasite may be part of the real evidence. Validation and test images are only resized to 224 x 224 and normalised with ImageNet statistics.

---

## 2. Model card

### 2.1 Model and training
| Item | Value |
|---|---|
| Backbone | ResNet-18 pretrained on ImageNet, new 2-class head (11,177,538 parameters) |
| Input | 224 x 224 RGB, batch size 32 |
| Phase 1 | Head only: 1,026 trainable parameters (0.009%), Adam, learning rate 1e-3, 2 epochs |
| Phase 2 | Last block (layer4) plus head: 8,394,754 trainable parameters (75.1%), learning rates 1e-4 and 1e-3, 3 epochs |
| Selection | Best epoch of each phase chosen by validation macro-F1 (not accuracy) |
| Time | About 42 s per epoch, 208 s for all 5 epochs |

| Phase | Epoch | Validation macro-F1 |
|---|---|---|
| 1 (head only) | 1 / 2 | 0.866 / 0.846 (best 0.866) |
| 2 (fine-tuning) | 1 / 2 / 3 | 0.935 / 0.937 / 0.935 (best 0.937) |

Fine-tuning improved validation macro-F1 by 0.071 (0.866 to 0.937). The score then levelled off, so more epochs with the same settings would probably add little.

### 2.2 Results on the test set (used once)
3,335 cells from 28 patients: accuracy 0.970, **macro-F1 0.969**, AUC 0.994.

| Class (default 0.5 cut-off) | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Parasitized | 0.964 | 0.961 | 0.963 | 1,340 |
| Uninfected | 0.974 | 0.976 | 0.975 | 1,995 |

Confusion matrix at 0.5: 1,288 parasitized found, 52 missed, 48 false alarms, 1,947 uninfected correct. The weakest class is Parasitized.

### 2.3 Threshold for a target recall
The threshold 0.154 was chosen on the validation set for a recall of at least 0.95 (validation recall 0.951, specificity 0.934) and then applied once to the test set.

| Metric at threshold 0.154 | Value |
|---|---|
| TP / FN / TN / FP | 1,316 / 24 / 1,879 / 116 |
| Sensitivity | 1,316 / 1,340 = **0.982** (target of 0.95 reached) |
| Specificity | 1,879 / 1,995 = **0.942** |
| PPV at test prevalence (40.2%) | 1,316 / 1,432 = **0.919** |
| PPV at 5% prevalence | **0.471** |
| PPV at 1% prevalence | **0.146** |

Lowering the threshold from 0.5 to 0.154 cut missed infections from 52 to 24 but raised false alarms from 48 to 116. This is the right trade when a missed infection is the costly error.

**PPV working at 5% prevalence:** the false-positive rate is 1 - specificity = 116 / 1,995 = 0.0581. PPV = (0.982 x 0.05) / (0.982 x 0.05 + 0.0581 x 0.95) = 0.0491 / (0.0491 + 0.0552) = 0.0491 / 0.1043 = 0.471.

**In plain language:** sensitivity and specificity do not change with prevalence, but PPV does. In a real screening sample most cells are uninfected, so even a 5.8% false-positive rate produces more false alarms than true positives, and about half of the positive calls at 5% prevalence would be wrong.

### 2.4 Grad-CAM shortcut check
Layer4 Grad-CAM on three correct and three wrong test predictions (patients C78, C184, C231, C171, C60, C6), plus a background-attention check on 300 random test images.

| | Share of pixels that is black background | Share of attention on the background |
|---|---|---|
| Parasitized | 0.272 | 0.122 |
| Uninfected | 0.262 | 0.164 |
| Overall | 0.266 | 0.146 |

- The background gets less attention than its share of the pixels in both classes, which argues against a strong background shortcut.
- The correct Parasitized heatmap for patient C184 is centred on a dark purple parasite spot, which is legitimate evidence.
- Four of the six heatmaps (patients C78, C171, C60 and C6) put their strongest heat on the cell edge, including all three wrong predictions, and Uninfected predictions put more attention on the background than Parasitized ones (0.164 against 0.122).

**Verdict: shortcut suspected (mild), not confirmed.** Six heatmaps and 300 attention scores are a small sample. **Follow-up experiment:** black out a ring around the cell outline (the outer 10% of the cell), re-score at threshold 0.154, and repeat with grayscale images to test the stain colour. A sharp drop in sensitivity (0.982) or specificity (0.942) would confirm a shortcut.

### 2.5 Limitations
- One data source: one hospital in Bangladesh, one parasite species (P. falciparum), Giemsa-stained thin smears.
- The cells were already cut out of the slides, so the model has never had to find them on a full slide.
- Only 200 patients, and only 28 patients in the test set, so the results carry real uncertainty.
- The test set is 40.2% parasitized, far above real screening prevalence.
- **Not for** diagnosing patients without confirmation by a trained microscopist.

---

## 3. Responsibility questions

**Who could be harmed, and how?** Patients. A missed infection (the model still misses 24 of 1,340 parasitized test cells at threshold 0.154) could delay treatment, and a false alarm (116 of 1,995 uninfected test cells) causes unnecessary worry and drug use. At realistic prevalence about half of the positive calls are false alarms, so a positive result is not a diagnosis.

**Where was the model not validated?** On other hospitals, microscopes, cameras and staining methods; on other parasite species such as P. vivax; on whole slides, where cells must first be found and cut out; on low-quality or dirty slides; and at low prevalence, where the PPV was calculated and never measured.

**What is the human role?** A trained microscopist reviews every positive result and a sample of negatives, and makes the diagnosis. The model is a screening aid that sorts cells for review and does not replace the person.

**What evidence would change my conclusions?** Results on patients and slides from other hospitals with similar sensitivity and specificity would raise my trust. A large drop when I black out the cell edge or convert images to grayscale would confirm a shortcut. A real screening sample with its true prevalence would show whether the PPV of 0.471 at 5% holds.

---

## 4. File

- **Week 6 notebook in this repo:** [week-6-malaria-detection.ipynb](week-6-malaria-detection.ipynb): full notebook with outputs and written answers (it was run on Kaggle, so its `/kaggle/input/` data paths only work there)

