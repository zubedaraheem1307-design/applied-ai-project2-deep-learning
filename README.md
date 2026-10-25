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

## File
- **Notebook in this repo:** [Week5_Transfer_Learning.ipynb](week-5-transfer-learning.ipynb): full notebook with outputs and written answers

- > **Update (Week 6):** I changed the project dataset. Instead of the shared chest X-ray reference dataset, I chose my own domain, as the lab encourages. The final project is malaria parasite detection in single blood-cell images, using the Cell Images for Detecting Malaria dataset (US National Institutes of Health, hosted on Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria): 27,558 images in 2 balanced classes (Parasitized and Uninfected, 13,779 each), from 200 patients. The method is unchanged: ResNet-18 transfer learning, first training only the head and then fine-tuning the last block, with the best epoch chosen by macro-F1. Because the classes are balanced, the main risk is not imbalance but patient leakage, so I split the data by patient and report macro-F1, sensitivity, specificity and PPV. The final results, data audit and Grad-CAM check are in the Week 6 section further down this README. It is still an educational project and not a diagnostic tool.
