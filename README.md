<div align="center">

# 📝 On the Generalization of Handwritten Text Recognition Models

📄 **Paper**: [ArXiv Link](https://arxiv.org/html/2411.17332v1)  
💻 **Code**: This repository  
✍️ **Authors**: Carlos Garrido-Munoz and Jorge Calvo-Zaragoza  
🏆 **Conference**: Accepted at CVPR 2025!  

</div>

---

## 🚀 Introduction

Handwritten Text Recognition (HTR) has achieved remarkable success under in-distribution (ID) conditions. However, **real-world applications require models to generalize to unseen domains**—out-of-distribution (OOD) settings.  

This repository provides the **official implementation of our CVPR 2025 paper**, where we conduct a **large-scale study** of OOD generalization in HTR models. Our research evaluates **336 OOD cases**, covering **8 state-of-the-art HTR models** across **7 datasets** in **5 languages**.

### 🔑 Key Findings:
- **Textual divergence** is the dominant factor limiting generalization, followed by **visual divergence**.
- No existing HTR model is explicitly designed for **robust OOD generalization**.
- **Synthetic data can improve OOD generalization**, but effectiveness depends on the architecture.
- We introduce **proxy metrics** that reliably estimate generalization error in OOD scenarios.
---

## 📂 Repository Structure 

```plaintext
📁 htr_ood/
│── 📂 src/             # Core implementation
│── 📂 models/          # Pre-trained and baseline models
│── 📂 datasets/        # Data preparation scripts
│── 📂 evaluation/      # OOD analysis and generalization metrics
│── 📜 requirements.txt # Required dependencies
│── 📜 README.md        # Project documentation
│── 📜 LICENSE          # License information
│── 📜 train.py         # Training script
│── 📜 test.py          # Model evaluation script
│── 📜 ood_analysis.py  # Out-of-distribution analysis
```

---

## 📦 Installation

### 🔧 Requirements
- Python 3.8+
- PyTorch >= 2.0
- NumPy, OpenCV, Matplotlib, Pandas
- WandB (optional for logging)

To install dependencies, run:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Models
We examined the following start-of-the-art models in the literature of HTR:
| **Model**       | **Citation** | **Config File**  | **Architecture**                           | **Alignment** | **Parameters** | **Input Size (H × W)** |
|----------------|------------|--------------|--------------------------------|--------------|----------------|----------------|
| **CRNN**       | Puigcerver et al., 2017       | `configs/model/crnn_puig.yaml`  | CRNN + CTC                     | CTC          | 9.6M           | **128 × 1024** |
| **VAN**        | Coquenet et al., 2020       | `configs/model/van_coquenet.yaml`   | Fully Convolutional Network (FCN) w. CTC | CTC       | 2.7M           | **64 × 1024**  |
| **C-SAN**      | Arce et al., 2022       | `configs/model/cnn_san_arce.yaml`  | CNN + Self Attention + CTC     | CTC          | 1.7M           | **128 × 1024** |
| **HTR-VIT**    | L et al., 2025       | `configs/model/htr_vit.yaml` | CNN + Vision Transformer + CTC | CTC          | 53.5M          | **64 × 512**   |
| **Kang**       | Kang et al., 2020       | `configs/model/transformer_kang.yaml`  | ResNet + Transformer           | Seq2Seq      | 90M            | **64 × 2227**  |
| **Michael**    | Michael et al., 2019       | `configs/model/crnn_michael.yaml` | CRNN + Attention Decoder       | Hybrid       | 5M             | **64 × 1024**  |
| **LT**         | Barrere et al., 2022       |`configs/model/light_barrere.yaml`    | CNN + Transformer + CTC        | Hybrid       | 7.7M           | **128 × 1024** |
| **VLT**        | Barrere et al., 2024       | `configs/model/v_light_barrere.yaml`   | CNN + Transformer + CTC        | Hybrid       | 5.6M           | **128 × 1024** |



---

## 📊 Datasets

We evaluate generalization performance on the following handwritten text datasets at the line-level:

| Dataset     | Language  | Period     | Writers | Configuration |
|------------|----------|------------- |---------|---------------|
| IAM        | English  | 1999         | 657     | iam            |
| Rimes      | French   | 2011         | 1.3K    | rimes          |
| Bentham    | English  | 18-19th c.   | 1       | bentham        |
| St. Gall   | Latin    | 9-12th c.    | 1       | saint_gall     |
| G.Washington | English | 1755        | 1       | washington     |
| Rodrigo    | Spanish  | 1545         | 1       | rodrigo        |
| ICFHR2016  | German   | 15-19th c.   | Unknown | icfhr_2016     |

To download and preprocess datasets, use:

```bash
python datasets/prepare_data.py --dataset IAM
```

---

## 🏗️ Training

To train an HTR model from scratch:

```bash
python src/train_crnn_ctc.py \
data/train/train_config/datasets=[iam] \
data.train.train_config.img_size=[64,1024] \
data.train.train_config.batch_size=16 \
data.train.train_config.binarize=True \
data.train.train_config.num_workers=8 \
trainer.max_epochs=500 \
trainer.deterministic=False \
model=crnn_puig \
tokenizer=tokenizers/char_tokenizer \
callbacks.early_stopping.patience=100 \
callbacks.model_checkpoint_base.filename=crnn_michael_src_iam \
callbacks.model_checkpoint_id.filename=\${callbacks.model_checkpoint_base.filename}_ID \
callbacks/heldout_targets=[rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
callbacks/optim_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
logger.wandb.offline=False \
logger.wandb.name=crnn_michael_src_iam \
train=True 
```

### 📜 Explanation of Parameters

| Parameter | Description |
|-----------|------------|
| `data/train/train_config/datasets=[iam]` | Specifies the dataset used for training (`IAM` in this case). |
| `data/val/val_config/datasets=[iam,rimes,...]` | Specifies the dataset used for validation (`IAM` in this case). If not specified, by default all datasets will be used for validation and testing. 
| `data/test/test_config/datasets=[iam,rimes,...]` | Specifies the dataset used for testing (`IAM` in this case). If not specified, by default all datasets will be used for validation and testing. 
| `data.train.train_config.img_size=[64,1024]` | Defines the input image size (height = 64, width = 1024). Depends on the architecture used!|
| `data.train.train_config.batch_size=16` | Sets the batch size to **16**. |
| `data.train.train_config.binarize=True` | Enables binarization of images for preprocessing. By default is true! |
| `data.train.train_config.num_workers=8` | Uses **8 worker threads** for data loading. |
| `trainer.max_epochs=500` | Limits training to **500 epochs**. |
| `trainer.deterministic=False` | Allows **non-deterministic** behavior. Should be turned off when training with the CTC objective!|
| `model=crnn_puig` | Uses the **CRNN+CTC model** from Puigcerver, 2017  for training. |
| `tokenizer=tokenizers/char_tokenizer` | Specifies the **character-level tokenizer** for text processing. |
| `callbacks.early_stopping.patience=100` | Implements early stopping if validation does not improve for **100 epochs**. |
| `callbacks.model_checkpoint_base.filename=crnn_puig_src_iam` | Defines the base filename for saving checkpoints. |
| `callbacks/heldout_targets=[rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016]` | Specifies datasets **not seen during training** for out-of-distribution (OOD) evaluation. This will create the N checkpoints optimized using a leave-one-out for later testing on the excluded. |
| `callbacks/optim_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016]` | Lists datasets used for **optimization and tuning**. This will create N checkpoints, each one optimized for the target dataset. |
| `logger.wandb.offline=False` | Enables **online** tracking using **Weights & Biases (WandB)**. |
| `logger.wandb.name=crnn_puig_src_iam` | Sets the WandB experiment name to **crnn_michael_src_iam**. |
| `train=True` | ** Sets the model in training mode 
Notes: all the checkpoints will be created by default in a folder checkpoints/
---

## 🧪 Evaluation

To evaluate a trained model on an unseen dataset:

```bash
python test.py --model crnn --dataset Rodrigo
```

For a complete OOD generalization analysis:

```bash
python ood_analysis.py
```

---

## 📈 Results

### 🔥 Key Observations:
- **CTC-based models** perform slightly better in OOD scenarios compared to Seq2Seq models.
- The **VAN model** shows the best generalization but only **outperforms others by ~1%**.
- **Hybrid models** struggle the most in generalization.
- **Using synthetic data improves OOD performance**, but choosing the right model is crucial.

#### 🎯 Average Character Error Rate (CER) in OOD Scenarios:

| Model       | IAM   | Rimes | Bentham | St. Gall | Rodrigo | Avg. CER (%) |
|------------|------|------|--------|---------|--------|-------------|
| CRNN       | 34.9 | 25.0 | 25.3   | 33.6    | 40.9   | 38.5        |
| VAN        | 28.6 | 21.3 | 26.6   | 39.8    | 38.5   | **37.4**    |
| HTR-ViT    | 33.7 | 28.3 | 33.3   | 36.5    | 38.5   | 41.2        |
| Michael    | 49.1 | 35.5 | 43.5   | 55.3    | 65.3   | 53.9        |

For full results, refer to our **[paper](https://arxiv.org/html/2411.17332v1)**.

---

## 🔮 Future Work

We identified **textual divergence** as the main challenge for OOD generalization in HTR models. Future research should:
- Design **HTR architectures explicitly optimized for OOD generalization**.
- Improve **synthetic-to-real adaptation** techniques.
- Investigate **pretraining strategies for cross-lingual generalization**.

---

## 📜 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{Garrido2025HTR,
  author    = {Carlos Garrido-Muñoz and Jorge Calvo-Zaragoza},
  title     = {On the Generalization of Handwritten Text Recognition Models},
  booktitle = {CVPR},
  year      = {2025}
}
```

---

## 🤝 Acknowledgments

This research was supported by **[Your Institution]**.  
We thank the **HTR community** for their contributions to open-source datasets.

---

## 📝 License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## 📬 Contact

For any questions, feel free to **open an issue** or reach out via **[Your Email/Website]**.

---

✨ *Star this repository if you find it helpful!* ⭐

