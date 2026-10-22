# When Labels Are Scarce: Fine-Tuned vs Frozen CLIP for Steel Surface Defect Classification

Coursework for **MAAI901 Deep Learning for Visual Data** (University of Wollongong in Dubai, Dr. Milan Dordevic) - Applied Vision Transformer Report.
Author: Karthik Krishna (ID 1697286).

**Question.** How many labelled images per class does a vision transformer need to classify steel surface defects, and is full fine-tuning worth its cost over a frozen encoder with a linear classifier? Beyond accuracy, we test calibration, rare defects, never-seen defect types and image corruptions.

## Headline results (NEU-CLS, 5-fold cross-validation)

Test accuracy (%), mean over 5 folds (1,800 test predictions per setting; k = labelled images per class):

| Method | k=1 | k=2 | k=5 | k=10 | k=50 | k=240 |
|---|---|---|---|---|---|---|
| Frozen CLIP + logistic regression (C=10) | 73.6 | 82.4 | 90.5 | 94.4 | 98.3 | 99.4 |
| Frozen CLIP + logistic regression (C tuned) | 74.6 | 84.3 | 92.7 | 96.5 | 98.7 | 99.4 |
| Partial fine-tune (CLIP, last 2 blocks) | 82.3 | 89.4 | 97.3 | 98.7 | 99.2 | 99.9 |
| **Full fine-tune (CLIP ViT-B/16)** | **87.3** | **92.4** | 97.1 | **99.4** | **99.7** | **100.0** |
| Full fine-tune (ImageNet-21k ViT-B/16) | 79.0 | 89.0 | 95.6 | 98.8 | 99.7 | 99.9 |

* Full fine-tuning leads a *tuned* frozen classifier by 12.7 points at k=1 (95% CI 8.4 to 17.1), 4.5 at k=5, 1.4 at k=20; from k=50 the interval includes zero.
* 95% accuracy needs 3 labels per class (full fine-tune) vs 10 (tuned frozen).
* Rare defect with 1 labelled example: class-weighted fine-tuning recovers 69% of its test images; plain fine-tuning 6%.
* Detecting a never-seen defect type (AUROC): fine-tuned CLIP 0.963, frozen 0.866.
* 22% of the fully fine-tuned model's errors at k=1 are made with probability >= 0.9.
* Blur/noise augmentation lifts accuracy at the heaviest blur from 61% to 97%, but only for distortions seen in training.
* Zero-shot CLIP is at chance (19.7% vs 16.7% for guessing).

The dataset is very homogeneous (every test image has a near-duplicate in the training pool; cosine similarity 0.921 to 0.995, median 0.980), so accuracy saturates above about 50 labels. Treat results as evidence about *this* benchmark. See the report for limitations.

## Repository layout

```
notebooks/NEU_ViT_experiment_v5.ipynb   the experiment suite as a notebook
src/neu_vit_experiments.py              the same code as a script (cells marked with "# %%")
src/analyze_results.py                  prints all summary tables and paired intervals used in the report
src/make_figures.py                     regenerates the report figures from the CSVs
results/full_run/                       raw outputs of the final run (CSVs, per-image predictions, config.json)
figures/                                figures used in the report and slides
report/                                 LaTeX source (Overleaf-ready) and compiled PDF
slides/pitch.pptx                       7-slide pitch deck
MODEL_CARD.md                           intended use, data, evaluation and limits
```

## Reproducing the results

1. Python 3.10 or newer. `pip install -r requirements.txt`.
2. The data are downloaded automatically from a public Kaggle copy of the NEU surface defect database with `kagglehub` (no credentials were needed in our run). To use your own copy, set `DATA_DIR` to a folder with the images. The notebook checks for exactly 1,800 images, 300 per class. **The dataset is not redistributed here.**
3. Run the script (or the notebook top to bottom):

```bash
# full profile: 5 folds, 8 label budgets, 2 seeds for k<=20, ImageNet ViT reference
PROFILE=full MPLBACKEND=Agg python src/neu_vit_experiments.py
# lighter alternatives
PROFILE=lite  ...    # 3 folds, 1 seed
PROFILE=local ...   # 2 folds, budgets up to 50, no ImageNet ViT (laptop-friendly)
SMOKE=1 ...         # a few-minute code check on tiny settings
```

Outputs go to `vit_results_v5/` (override with `OUT_DIR`) and are saved after every step; **re-running resumes from the saved CSVs**. A zip of the outputs is written at the end.

Device selection is automatic: CUDA, then Apple MPS, then CPU. CPU is far too slow for the fine-tuning experiments.

Then regenerate the tables and figures:

```bash
python src/analyze_results.py results/full_run
python src/make_figures.py results/full_run figures
```

### Our run
Apple M4 Air (MPS), Python 3.13, PyTorch 2.14.1, roughly 27 hours wall-clock for the full profile including pauses. On a Mac we found that importing scikit-learn before PyTorch under conda crashed the kernel (duplicate OpenMP libraries); the script imports PyTorch first and sets `KMP_DUPLICATE_LIB_OK`. A clean virtual environment avoided the problem entirely.

## Experimental protocol in brief

* **Data:** NEU-CLS, 1,800 images of 200x200 pixels, six defect types (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches), 300 per class. Resized to 224x224 (bicubic) and normalised per backbone.
* **Splits:** 5 folds; each holds out 60 images per class for testing. k in {1, 2, 3, 5, 10, 20, 50, 240} training images per class are drawn from the other 240 per class (two draws for k<=20). The tuned frozen classifier selects its regularisation on 60 separate pool images per class.
* **Hyper-parameters** (AdamW, encoder lr 1e-5, head lr 1e-3, batch 32, at least 100 steps, cosine schedule) were fixed before any test result was seen. Experiment 5 checks sensitivity to them.
* **Metrics:** accuracy, macro-F1, log-loss, expected calibration error, confident errors, AUROC, rare-class recall and precision. Differences are paired across folds with 95% t-intervals (approximate, because folds share training images).

| Experiment | Output file |
|---|---|
| 1 Learning curves, 5-fold | `exp1_learning_curves.csv`, `predictions.pkl` |
| 2 One rare defect | `exp2_rare_defect.csv` |
| 3 Never-seen defect (open set) | `exp3_open_set.csv` |
| 4 Robustness (blur, noise, brightness, contrast; with and without distortion augmentation) | `exp4_robustness.csv` (one fold, k=20 and 240) |
| 5 Learning-rate and regularisation sensitivity | `exp5_sensitivity.csv` (one fold) |
| 6 Zero-shot prompts | `exp6_zero_shot.csv`, `exp6_zero_shot_confusion.csv` |
| Near-duplicate analysis | `nn_similarity.npz`; figures in `figures/` |

## Data, licence and ethics notes

* NEU surface defect database: please cite K. Song and Y. Yan, *Applied Surface Science* 285 (2013) 858-864, and Y. He, K. Song, Q. Meng and Y. Yan, *IEEE Trans. Instrum. Meas.* 69(4) (2020) 1493-1504. The official page asks for citation and states no formal licence; this repository uses the data for coursework only and does not redistribute it.
* No images contain people. Pretrained weights (OpenAI CLIP via OpenCLIP, ImageNet-21k ViT via timm) are downloaded at run time and are not included.
* Models here are research prototypes, **not** validated for production quality control. See `MODEL_CARD.md`.

Code is released under the MIT licence (see `LICENSE`).
