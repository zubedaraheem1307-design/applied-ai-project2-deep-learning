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


## Week 7: Text and Multimodal AI

- Text classifier: SMS Spam Collection (UCI), 5,572 texts, macro-F1 **0.974** (word 1-2 grams) vs **0.982** (char 2-5 grams); spam precision/recall 0.979/0.933 (word) vs 0.986/0.953 (char)
- Top positive / negative features: call, txt, free, text, reply / my, me, ok, that, it (some look like dataset shortcuts: uk, gt, lt, da)
- NER on my domain (malaria diagnosis and medical labs, 10 sentences, 36 gold entities): out-of-the-box held-out F1 **0.778** -> with entity ruler **0.944** (rule-writing sentences 0.545 -> 1.000, inflated because the rules were written from them); lab sentences 0.533 -> 0.875
- Regex fields extracted: slide ID, body temperature, haemoglobin, date, amount, mobile, CNIC, email, invoice number; false positives tested: 7 negative cases (e.g. "Room temperature 25 °C", "SL-12345", "Hb 9.88 g/dL"), all rejected; 12/12 tests passed
- OCR: CER raw **0.143** -> after correction rules **0.000** (date "3O/09/2O26"); lab pairs 0.062 and 0.059 -> 0.000; EasyOCR on a clean synthetic invoice 0.000 -> 0.000
- Pipeline: image model + text evidence -> JSON record (example below)
- Privacy: CNIC, phone, and email masked before logging (checked: no phone or CNIC digits in the JSON)

### Example case record (printed by my notebook)

```json
{
  "image": "C100P61ThinF_IMG_20150918_144104_cell_162.png",
  "image_probabilities": {
    "Parasitized": 1.0,
    "Uninfected": 0.0
  },
  "note_masked": "Patient Alisha Noor, aged 24, has fever with chills for 4 days after travel. Blood smear taken at PIMS Islamabad on 12/10/2026. Contact [PHONE], CNIC [CNIC].",
  "text_evidence": {
    "entities": [
      ["Alisha Noor", "PERSON"],
      ["24", "DATE"],
      ["4 days", "DATE"],
      ["PIMS", "ORG"],
      ["12/10/2026", "DATE"],
      ["CNIC", "ORG"],
      ["CNIC", "ORG"]
    ],
    "fields": {
      "date": ["12/10/2026"]
    },
    "urgent_terms": ["fever", "chills"]
  },
  "decision": "REFER",
  "reasons": [
    "image model: P(Parasitized) = 1.00",
    "note mentions: fever, chills"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

The record keeps its wrong entities on purpose: "24" (an age) is tagged DATE, "CNIC" is tagged ORG, and Islamabad is missing. They don't affect the decision, which reads only the image probability and the urgent terms.

**Limitations:** small NER gold set (10 sentences), synthetic OCR sample, the image model looks at one cropped cell, and the system is decision support only, not a diagnosis.


Week 8: Ship it (FastAPI service for the malaria classifier)
Week 8 turns the Project 2 system into a tested web service. It serves my Week 6 image model (ResNet-18, classes `Parasitized` / `Uninfected`) and my Week 7 text evidence rules. It classifies a blood-cell image, masks private details in a clinical note, flags urgent words, and returns a `REFER` or `ROUTINE` decision for a human to review. All code is in the `api` folder.
> Decision support only. A qualified human decides.
Files
File	Purpose
`api/app.py`	FastAPI service with `/health`, `/predict`, `/analyze`
`api/test_api.py`	8 automated tests
`api/bench.py`	Latency benchmark against a running server
`api/sample.jpeg`	Test image used by the tests and the benchmark
`api/README.md`	Same documentation, kept next to the code
Endpoints
Method	Path	Purpose
GET	`/health`	Shows the model is loaded, the classes, and the positive class
POST	`/predict`	Upload one JPEG/PNG image, get label, probabilities and latency
POST	`/analyze`	Upload an image plus a text note, get a full case record
Settings (environment variables): `MODEL_PATH` (default `week6_model.pt`), `POSITIVE` (default `Parasitized`, the class that must not be missed), `THRESHOLD` (default `0.5`), `API_KEY` (optional).
Model details
Checkpoint: `week6_model.pt` (about 43 MB, saved in Week 6). It is not committed to this repository because of its size; place it in the `api` folder before running.
Backbone: ResNet-18 (transfer learning), 224 x 224 RGB input, ImageNet normalisation. The service rebuilds the network and the preprocessing from the settings stored in the checkpoint, so the API and training use the same transform.
Week 6 test-set result (3,335 images): macro-F1 0.969, accuracy 0.970. Parasitized: precision (PPV) 0.964, recall (sensitivity) 0.961. Uninfected: precision 0.974, recall (specificity) 0.976. Confusion matrix (rows = true, columns = predicted, order Parasitized, Uninfected): [[1288, 52], [48, 1947]], so 52 infected cells were missed and 48 healthy cells were flagged. The test set is about 40% Parasitized, so the PPV would be lower in a setting where infection is rarer.
Urgent words checked in notes: fever, chills, vomiting, breath, confusion, jaundice, seizure, weakness.
How to run
```
cd api
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pillow pytest httpx numpy
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000/docs and use "Try it out". On Mac/Linux, activate with `source .venv/bin/activate`.
API usage
```
curl -X POST http://127.0.0.1:8000/predict ^
     -F "file=@sample.jpeg;type=image/jpeg"

curl -X POST http://127.0.0.1:8000/analyze ^
     -F "file=@sample.jpeg;type=image/jpeg" ^
     -F "note=Fever and chills since Monday. Guardian 0300-1234567, CNIC 35202-1234567-1"
```
(On Mac/Linux use `\` instead of `^`.)
Results I observed on my sample image:
`/health`: `status: ok`, `model_loaded: true`, classes `["Parasitized", "Uninfected"]`, positive `Parasitized`.
`/predict`: label `Parasitized`, probability about 0.9999 for Parasitized.
`/analyze`: `note_masked` shows `[PHONE]` and `[CNIC]` instead of the real numbers, `urgent_terms` contains `fever` and `chills`, and `decision` is `REFER`. The digits `1234567` do not appear anywhere in the response.
Error handling
Bad input	Status	Response
A non-image file (`app.py` sent as `text/plain`)	415	`{"detail":"Upload a JPEG or PNG image"}`
An image larger than 5 MB	413	`{"detail":"Image larger than 5 MB"}`
A file that is not a readable image	400	`{"detail":"File is not a readable image"}`
Missing file	422	validation error from FastAPI
A note longer than 2,000 characters	422	`{"detail":"Note longer than 2,000 characters"}`
One bad input beyond the tests: I sent a 2,001-character note to `/analyze`. The API returned `{"detail":"Note longer than 2,000 characters"}` with status 422, and no stack trace. I also sent a 6 MB file to `/predict` and got `{"detail":"Image larger than 5 MB"}`.
Tests
```
pytest -q test_api.py
........                                  [100%]
8 passed, 1 warning in 32.30s
```
The 8 tests cover: health, response schema and probabilities summing to 1, parity between the API and calling the model directly, wrong file type (415), too-large file (413), corrupt image (400), missing file (422), and PII masking in `/analyze` (no part of the phone number or CNIC appears in the response). The single warning is a deprecation notice from a library and does not affect the results.
Latency and capacity
Benchmark: 50 requests to `/predict` against a running server (`python bench.py`).
Item	Value
Hardware	Intel Core i5-6300U @ 2.40 GHz (2 cores), 7.9 GB RAM, CPU only
p50	271 ms
p95	660 ms
max	1343 ms
Throughput, 1 worker	about 3.0 requests/s
Little's law. L = λW with L = 1 for one worker and W = p50 = 0.271 s, so λ = 1 / 0.271 ≈ 3.7 requests/s. Over a day that is about 3.7 x 86,400 ≈ 319,000 requests per day if traffic were perfectly even. At the measured 3.0 requests/s it is about 259,000 per day. Real traffic comes in bursts, so safe capacity is lower.
Check Your Understanding
A client uploads a PDF to /predict. Which status code, and why not 500? 415 Unsupported Media Type. The client sent the wrong kind of file, which is a client error (4xx). A 500 means the server crashed from a bug and could leak a stack trace. The API checks the content type first and returns a clean message.
Why is the model loaded in lifespan rather than inside the endpoint? Loading the checkpoint and rebuilding the network takes seconds. In `lifespan` it happens once at startup, so each request only runs the fast forward pass. Loading inside the endpoint would add that delay to every request and waste memory.
p50 = 40 ms and p95 = 400 ms. What does that tell you, and what would you check? Most requests are fast, but about 1 in 20 is ten times slower: a tail-latency problem that the average hides. I would check for cold starts, background processes or CPU throttling, blocking of the event loop, requests queuing together, and unusually large inputs. In my own run (p50 271 ms, p95 660 ms) the likely causes are a low-power 2-core laptop CPU and other programs running at the same time.
500,000 requests a day, peak 5 times the average, 25 requests per second per worker. How many workers? Average = 500,000 / 86,400 ≈ 5.8 requests/s. Peak = 5 x 5.8 ≈ 29 requests/s. Workers = 29 / 25 ≈ 1.16, so 2 workers.
Why can model inference inside an async def endpoint slow down other requests? An async server runs on one event loop that switches between requests only at an `await`. Model inference is heavy CPU work with no `await`, so the loop is blocked and other requests, even `/health`, wait. `run_in_threadpool` moves the work to a thread so the loop stays free.
How many communication channels does a 4-person team have? n(n - 1) / 2 = 4 x 3 / 2 = 6.
Privacy
`/analyze` masks CNIC numbers and mobile numbers before any other processing, so the original digits never appear in the response. Masking runs first, and a test checks the whole response for leaks.
Reflection
Building this API showed me that a model that works in a notebook is only the start of a working service. My first PyTorch install timed out on a slow connection, and I fixed it by retrying with a longer timeout and more retries. Later, an indentation error stopped the server from starting after I saved `app.py` in the wrong format, which taught me that the server needs the file as exact plain text. The benchmark surprised me most: on my i5-6300U laptop the median was 271 ms but p95 was 660 ms, so the average hides slow requests and capacity planning should use the tail, not the mean. Testing bad inputs (a text file, a 6 MB file and a 2,001-character note) was reassuring, because each got a clean error code with no stack trace, and the PII test showed that the phone number and CNIC never appear in a response. If I continued this project, I would add an API key, deploy it publicly, and tune the threshold so that Parasitized cases are rarely missed.
Limits
This is a student project and a decision-support demo. It must not be used to diagnose patients or replace a laboratory professional.
