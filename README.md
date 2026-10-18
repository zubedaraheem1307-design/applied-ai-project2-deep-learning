# Project 2: Deep Learning Applications

**Name:** Zubeda Raheem
**Subject:** Introduction to Applied AI

## Project 2, Week 5: Scene Classification with Transfer Learning
- Dataset: Intel Image Classification, 6 classes, 14,034 train (11,929 train + 2,105 validation) / 3,000 test
- Model: ResNet-18 pretrained on ImageNet
- Feature extraction (3,078 trainable parameters): val accuracy 0.901
- Fine-tuning layer4 (about 8.4M trainable): val accuracy 0.931
- Test accuracy (used once): 0.935; weakest class: glacier (recall 0.870), confused with mountain
- Most instructive error: A railway hall with tracks, labelled "buildings", was predicted as "street" with only 0.53 confidence, which shows that some labels fit neither class well.
- Next: my own domain in Week 6: chest X-ray classification (Normal vs Pneumonia), see the proposal below

## Week 6 Proposal
**Problem:** Pneumonia is a serious lung infection in children, and diagnosing it from chest X-rays depends on scarce expert radiologists. For Project 2, I will build an image classifier that labels a chest X-ray as Normal or Pneumonia. I will use ResNet-18 transfer learning with the same pipeline as Week 5: augmentation on training data only, feature extraction, then fine-tuning, and a single final test. Because pneumonia images outnumber normal ones about 3 to 1, I will report per-class recall and the confusion matrix, not just accuracy. **Who benefits:** clinics with few radiologists, as a second-opinion screening aid. This is an educational project, not a diagnostic tool. **Dataset:** Chest X-Ray Images (Pneumonia) by Paul Mooney on Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), 5,863 JPEG images in 2 classes (Normal and Pneumonia), licensed CC BY 4.0. Source: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell, 2018.
