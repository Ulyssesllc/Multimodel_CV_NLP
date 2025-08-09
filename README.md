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

2. **GLAMI-1M Experiment (in `GLAMI-1M/` directory)**
   - Contains exploratory scripts using Vision Transformers, contrastive learning, and attention-based fusion on a larger dataset.
   - Scripts included but not part of the core Amazon workflow:
     - `Contrastive.py`, `attention_model.py`, `train.py`, `MoE.py`, `Q_former.py`, `Q_bottleneck.py`, `single.py`, `solve_problem.py`.

---

Each folder has its own README with details on data location, dependencies, and how to run training.
