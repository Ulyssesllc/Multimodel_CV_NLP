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
     4. Run the training script:
        ```bash
        python ML.py
        ```
        - This will train the `FusionModule` on the dataset and print loss per epoch.
        - The final model weights are saved to `fusion_model.pth` in the same folder.
     5. (Optional) To use the trained model for inference or fine-tuning, load `fusion_model.pth` in your script.

2. **GLAMI-1M Experiment (in `GLAMI-1M/` directory)**
   - Contains exploratory scripts using Vision Transformers, contrastive learning, and attention-based fusion on a larger dataset.
   - Scripts included but not part of the core Amazon workflow:
     - `Contrastive.py`, `attention_model.py`, `train.py`, `MoE.py`, `Q_former.py`, `Q_bottleneck.py`, `single.py`, `solve_problem.py`.

---

Each folder has its own README with details on data location, dependencies, and how to run training.
