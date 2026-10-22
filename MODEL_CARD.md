# Model card: fine-tuned and frozen CLIP ViT-B/16 for NEU steel surface defect classification

Follows the structure suggested by Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). This card describes *experimental models* produced by the code in this repository. No trained weights are published; every model is retrained from pretrained checkpoints by running the code.

## Model details
* **Backbones:** CLIP ViT-B/16 (OpenAI weights through OpenCLIP, `ViT-B-16-quickgelu`) and, as a reference, `vit_base_patch16_224.augreg_in21k_ft_in1k` from timm.
* **Variants:** frozen encoder + logistic regression; partial fine-tuning (last 2 of 12 blocks, final norm and projection); full fine-tuning; zero-shot CLIP with text prompts.
* **Output:** one of six defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) with softmax probabilities.
* **Author:** Karthik Krishna, University of Wollongong in Dubai (coursework, October 2026). MIT-licensed code.

## Intended use
* **In scope:** studying how label budget, fine-tuning depth, class weighting and augmentation affect transformer classifiers on a small industrial image benchmark; teaching and discussion.
* **Out of scope:** automatic accept/reject decisions on production steel, safety-critical inspection, or any plant, camera or steel grade not represented in NEU-CLS, without local validation and human oversight.

## Data
NEU-CLS: 1,800 grayscale 200x200 images of hot-rolled steel strips, 300 per class, from one laboratory setup (Song and Yan 2013; He et al. 2020). The images are homogeneous: every test image has a near-duplicate in the training pool (cosine similarity of CLIP embeddings 0.921 to 0.995). No people are depicted. The data are not redistributed here.

## Evaluation
5-fold cross-validation, 60 test images per class per fold, k = 1 to 240 training images per class, up to 2 random draws. Accuracy, macro-F1, log-loss, expected calibration error, confident errors, rare-class recall, unseen-class AUROC, and accuracy under blur, noise, brightness and contrast shifts. Headline: full fine-tuning of CLIP reaches 95% accuracy with 3 labels per class; the advantage over a tuned frozen encoder disappears beyond about 50 labels. Full numbers in `results/full_run/` and the report.

## Limitations and risks
* **Saturation and homogeneity:** accuracy is near 100% with 50 or more labels; results do not show how models behave on other plants, cameras or steel grades.
* **Confident errors:** at 1 label per class, 22% of the fully fine-tuned model's errors are made with probability >= 0.9. High confidence must not authorise automatic release.
* **Weak classes:** scratches (recall 0.66), pitting (0.80) and inclusion (0.85) at 1 label per class.
* **Robustness:** clean-trained models degrade sharply under heavy blur and noise. Distortion augmentation fixes the trained distortions only and slightly hurt performance under strong brightening.
* **Zero-shot CLIP is at chance** on these images (19.7%).
* **Pretraining data:** CLIP's web-scale training data were not audited; effects on steel textures are unlikely but unmeasured.
* **Statistics:** 5 folds share training images, so intervals are approximate; the corruption and sensitivity tests use one fold; no convolutional baseline.

## Ethical considerations and recommendations
Keep a human in the loop for low-confidence and novel-looking items; set per-defect thresholds on local data; log decisions; revalidate on every deployment site; treat the system as an inspector's assistant rather than a replacement; if cameras capture people, apply data-protection law (for example the UAE federal data protection law, Decree-Law 45 of 2021); weigh the compute cost of fine-tuning against frozen features when labels are plentiful.
