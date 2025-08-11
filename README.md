# Multimodel_CV_NLP Repository

This repo contains two separate experiments for multimodal vision-and-language modeling:

1. **Amazon Dataset (in `amazon/` directory)**
   - Uses a small subset of product images and descriptions (`training_data.csv` + `amazon_dataset/`) to train a simple MLP fusion model.
   - Main scripts:
     - `process_data.py`: Dataset class for loading images and text.
     - `MLP.py`: Encoders and `MLP_fusion` model.
     - `ML.py`: Training loop (split train/test, DataLoader, optimizer, epochs).
   
   **Detailed Workflow:**
     1. Change into the `amazon/` directory:
        ```bash
        cd amazon
        ```
     2. Install required packages:
        ```bash
        pip install torch torchvision transformers scikit-learn pandas
        ```
     3. Verify that the following data is available:
        - `training_data.csv` (product metadata and labels)
        - `amazon_dataset/` directory containing image files
     4. Run training (advanced pipeline now with config + metrics + mixed precision):
        ```bash
        python ML.py
        ```
        - Features: dynamic class mapping, seeding, early stopping, LR scheduler (cosine), AMP, gradient clipping, accuracy metrics, checkpointing (`checkpoint_best.pth`), history log `training_history.json`.
        - Best model weights saved: `fusion_model_best.pth`.
     5. Resume training from checkpoint (optional):
        - Edit `CONFIG.resume_checkpoint` in `config.py` to point to `checkpoint_best.pth` or another checkpoint, then rerun `python ML.py`.
     6. Adjust hyperparameters:
        - Edit `amazon/config.py` (batch_size, epochs, lr, scheduler, dropout, freezing, backbone choice, mixed precision, etc.).
     7. Visualize & export VLM-style predictions:
        ```bash
        python visualize_results.py --model-path fusion_model_best.pth --num-samples 6 --top-k 5
        ```
        - Outputs:
          - `amazon/vlm_outputs/predictions_grid.png`
          - `amazon/vlm_outputs/predictions.json` (Top-K probs, descriptions, GT/Pred)
            - Enhanced single-file composite (all panels in one PNG):
               ```bash
               # Grid + composite
               python visualize_results.py --model-path fusion_model_best.pth --num-samples 6 --top-k 5 --composite

               # Only composite (skip matplotlib grid)
               python visualize_results.py --model-path fusion_model_best.pth --num-samples 8 --top-k 5 --composite --no-grid

               # Horizontal layout instead of vertical stacking inside composite
               python visualize_results.py --model-path fusion_model_best.pth --num-samples 5 --top-k 5 --composite --layout horizontal
               ```
               - Extra output: `amazon/vlm_outputs/predictions_composite.png`
               - Panel content: Sample index, GT (name+index), Pred (name+index), Top-K list, Description.

2. **GLAMI-1M Experiment (in `GLAMI-1M/` directory)**
   - Upgraded training workflow includes:
     - Central config: `GLAMI-1M/config.py` (batch_size, epochs, lr, scheduler, patience, AMP, grad clip, workers)
     - Refactored `train.py` with: seeding, early stopping on test accuracy, cosine/plateau scheduler, AMP, gradient clipping, JSON history logging, best checkpoint saving (`checkpoints/best_model.pth`).
     - Logs stored in `glami_logs/` (train_log.txt, history.json, attention_log.txt, attention_history.json).
   - Run contrastive fusion model:
     ```bash
     cd GLAMI-1M
     python train.py --model Q_cons_fusion
     ```
   - Run simple MLP fusion baseline:
     ```bash
     python train.py --model MLP_fusion
     ```
   - Adjust hyperparameters by editing `GLAMI-1M/config.py`.
   - Resume training: set `resume` path in config (future extension for loading full optimizer/scaler state).

### GLAMI-1M Dataset Preparation

Expected folder structure:
```
GLAMI-1M/
  train.py
  config.py
  prepare_glami.py
  Contrastive.py
  MLP.py
  ...
  images/               # all image files
  GLAMI-1M-train.csv    # stratified train split
  GLAMI-1M-test.csv     # stratified test split
```

Minimum CSV schema (both train & test):
```
img_path,category_name
000001.jpg,Dresses
000002.jpg,Shoes
...
```

Steps to build splits from a master metadata file `all_meta.csv`:
```bash
cd GLAMI-1M
python prepare_glami.py --meta all_meta.csv --images images --test-size 0.1 --seed 42 --summary summary.json
```
Outputs:
- GLAMI-1M-train.csv
- GLAMI-1M-test.csv
- summary.json (optional stats: counts, classes, missing files)

Utility flags:
- `--lowercase`: normalize filenames to lowercase.
- `--test-size 0.1`: set test ratio.
- `--seed`: reproducible split.

Validation logic removes rows whose `img_path` does not exist under `images/`.

Download & prepare directly from an official (e.g. test-only) ZIP subset:
```bash
cd GLAMI-1M
python prepare_glami.py \
   --meta GLAMI-1M-dataset--test-only.csv \
   --images images \
   --download-url https://huggingface.co/datasets/glami/glami-1m/resolve/main/GLAMI-1M-dataset--test-only.zip \
   --skip-split \
   --summary summary_test_only.json
```
Notes:
- Provide the correct CSV inside (or accompanying) the downloaded ZIP as --meta.
- Use `--skip-split` because a test-only subset should not be split again.
- Add `--force-download` to re-download if the zip already exists.

After preparation run training:
```bash
python train.py --model Q_cons_fusion
```

### GLAMI-1M Visualization

Once you have a trained checkpoint (e.g. `checkpoints/best_model.pth`):

Basic grid + JSON + (optional) composite for contrastive model:
```bash
cd GLAMI-1M
python visualize_glami.py --checkpoint checkpoints/best_model.pth --model-type Q_cons_fusion --num-samples 6 --top-k 5 --composite
```

MLP baseline visualization:
```bash
python visualize_glami.py --checkpoint checkpoints/best_model.pth --model-type MLP_fusion --num-samples 6 --top-k 5 --composite
```

Composite only (skip matplotlib grid) horizontal layout:
```bash
python visualize_glami.py --checkpoint checkpoints/best_model.pth --model-type Q_cons_fusion --num-samples 8 --top-k 5 --composite --no-grid --layout horizontal
```

Outputs (default `glami_vis/`):
- `glami_predictions_grid.png` (if not using `--no-grid`)
- `glami_predictions_composite.png` (when `--composite`)
- `glami_predictions.json`

Panel content mirrors Amazon: Sample id, GT (name+index), Pred (name+index), Top-K probability list, short description placeholder (currently category-based; extendable to richer metadata).

Flags summary:
- `--top-k`: number of probabilities to list.
- `--composite`: enable single-file stacked panels.
- `--layout horizontal|vertical`: composite arrangement.
- `--no-grid`: suppress matplotlib multi-axes grid.
- `--no-show`: headless mode.

If you retrain with different class counts the script adjusts final linear layer shape when loading state dict (non-strict load for safety).

### Data Sources Notes
- `training_data.csv` (Amazon) contains product image paths, textual descriptions, label text and numeric IDs.
- `amazon_dataset/` stores the corresponding image assets referenced by `training_data.csv`.
- `GLAMI-1M-train.csv` / `GLAMI-1M-test.csv` (generated) contain fashion product metadata for GLAMI-1M with at least `img_path` and `category_name`.
- All class indices are built dynamically at runtime (`label_map`) to avoid hardcoding.

Ensure you have rights / licenses for redistributed images. Large-scale runs should consider converting CSV to Parquet for faster loading.

---

Each folder has its own README with details on data location, dependencies, and how to run training.
