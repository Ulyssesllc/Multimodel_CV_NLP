# 🍜 Multimodal Food Classification with Text & Image (FOOD-101)

This repository explores multimodal deep learning techniques for food classification using the [FOOD-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), which includes **image** and **text** data. Our goal is to accurately classify food categories using a combination of visual and textual modalities.

---

## 📌 Problem Statement

Food classification is challenging due to high intra-class variation and subtle visual similarities across different food types. Traditional vision-only models can struggle in such cases. This project enhances classification performance by leveraging **multimodal inputs** — combining both **image** and **textual** information.

---

## 🧠 Approach

We explore and compare three different model architectures:

### 1. ✅ MLP Baseline
- Concatenates image and text features.
- Uses simple feed-forward layers to classify.
- Acts as a baseline for performance comparison.

### 2. 🔀 Cross-Attention Model
- Applies a Cross-Multihead Attention layer between learnable query tokens and image features.
- Inspired by the success of cross-modal attention in vision-language models.

### 3. 🚀 Refined Q-Former (Inspired by BLIP-2)
- A more advanced design based on the Q-Former architecture from BLIP-2.
- Learns a set of query tokens that attend to image features.
- Output is fused with textual features before classification.
- Includes residual connections, layer normalization, and deeper fusion MLP for better performance and stability.

---

## 📂 Dataset

- **FOOD-101**: 101 food categories with 1,000 images each.
- Textual information includes class names and optionally short descriptions.

