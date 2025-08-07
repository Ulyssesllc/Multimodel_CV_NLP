# Multimodel_CV_NLP Repository

This repo contains two separate experiments for multimodal vision-and-language modeling:

1. **Amazon Dataset (in `amazon/` directory)**
   - Uses a small subset of product images and descriptions (`training_data.csv` + `amazon_dataset/`) to train a simple MLP fusion model.
   - Main scripts:
     - `process_data.py`: Dataset class for loading images and text.
     - `MLP.py`: Encoders and `MLP_fusion` model.
     - `ML.py`: Training loop (split train/test, DataLoader, optimizer, epochs).
   
   **Detailed Workflow:**
     1. Navigate to the `amazon/` folder:
        ```bash
        cd amazon
        ```
     2. Install dependencies:
        ```bash
        pip install torch torchvision transformers scikit-learn pandas
        ```
     3. Ensure data files are present:
        - `training_data.csv` (metadata)
        - `amazon_dataset/` folder with product images
     4. Run training script:
        ```bash
        python ML.py
        ```
     5. Check console output for loss and accuracy over epochs.

2. **GLAMI-1M Experiment (in `GLAMI-1M/` directory)**
   - Contains exploratory scripts using Vision Transformers, contrastive learning, and attention-based fusion on a larger dataset.
   - Scripts included but not part of the core Amazon workflow:
     - `Contrastive.py`, `attention_model.py`, `train.py`, `MoE.py`, `Q_former.py`, `Q_bottleneck.py`, `single.py`, `solve_problem.py`.

---

Each folder has its own README with details on data location, dependencies, and how to run training.
