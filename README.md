<div align="center">

# 📝 On the Generalization of Handwritten Text Recognition Models (WIP)

📄 **Paper**: [ArXiv Link](https://arxiv.org/html/2411.17332v1)  
💻 **Code**: This repository  
✍️ **Authors**: Carlos Garrido-Munoz and Jorge Calvo-Zaragoza  
🏆 **Conference**: Accepted at CVPR 2025!  
m
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
│── 📂 configs/         # Configuration files
│── 📂 data/            # Data used for the experiments
│── 📂 docker/          # Contains Dockerfile
│── 📂 scripts/         # Scripts for launching complete experiments
│── 📂 src/             # Core implementation: modules, architectures, data reading, etc. 
│── 🐳 Dockerfile       # Dockerfile required for setting up the environment
|── 🔧 Makefile         # Utility task for cleaning, syncing and project maintenance
│── 📜 requirements.txt # Required dependencies
│── 📜 README.md        # Project documentation
│── 📜 LICENSE          # License information
```
This code uses the amazing [Hydra template repo](https://github.com/ashleve/lightning-hydra-template) from ashleve.

<!-- │── 📜 ood_analysis.py  # Out-of-distribution analysis -->
<!-- │── 📜 LICENSE          # License information -->

---

## 📦 Installation

### 🐳 Docker
You can set up a container by running:

```bash
bash scripts/start_container.sh [IMAGE_NAME] [GPU_DEVICE] [SHM_SIZE_GB]
```

This script will build and run a GPU Docker container with custom settings.

---

#### 🧾 Arguments

| Argument        | Description                                 | Default            | Required |
|-----------------|---------------------------------------------|--------------------|----------|
| `IMAGE_NAME`    | Name to assign to the Docker image          | `htr-ood-image`    | No       |
| `GPU_DEVICE`    | GPU ID to assign to the container           | `0`                | No       |
| `SHM_SIZE_GB`   | Shared memory size in GB                    | `24`               | No       |

---

#### 🧪 Examples

Run with default settings:

```bash
bash scripts/start_container.sh
```

Build and run using GPU 0, a custom image name and 20gb of shared memory:

```bash
bash scripts/start_container.sh htr-ood-image 0 20
```

Run with 48GB shared memory:

```bash
bash scripts/start_container.sh htr-ood-image 0 48
```

Then you can just execute the container:
```bash
docker exec -it htr-ood-image-container bash
```


OR, you can just set up your libraries by yourself installing the requirements: 

### 🔧 Requirements
- Python 3.8+
- PyTorch >= 2.0
- Wandb (optional for logging)
- NumPy, OpenCV, Pillow, etc. 
- Others (mostly Hydra stuff)

To install dependencies, run:

```bash
pip install -r requirements.txt
```
---

## ⚙️ Models
We examined the following start-of-the-art models in the literature of HTR:
| **Model**       | **Citation** | **Config File (in configs/model/)**  | **Architecture**                           | **Alignment** | **Parameters** | **Input Size (H × W)** |
|----------------|------------|--------------|--------------------------------|--------------|----------------|----------------|
| **CRNN**       | Puigcerver et al., 2017       | `crnn_puig.yaml`  | CRNN + CTC                     | CTC          | 9.6M           | **128 × 1024** |
| **VAN**        | Coquenet et al., 2020       | `van_coquenet.yaml`   | Fully Convolutional Network (FCN) w. CTC | CTC       | 2.7M           | **64 × 1024**  |
| **C-SAN**      | Arce et al., 2022       | `cnn_san_arce.yaml`  | CNN + Self Attention + CTC     | CTC          | 1.7M           | **128 × 1024** |
| **HTR-VIT**    | Li et al., 2025       | `htr_vit.yaml` | CNN + Vision Transformer + CTC | CTC          | 53.5M          | **64 × 512**   |
| **Kang**       | Kang et al., 2020       | `transformer_kang.yaml`  | ResNet + Transformer           | Seq2Seq      | 90M            | **64 × 2227**  |
| **Michael**    | Michael et al., 2019       | `crnn_michael.yaml` | CRNN + Attention Decoder       | Hybrid       | 5M             | **64 × 1024**  |
| **LT**         | Barrere et al., 2022       |`light_barrere.yaml`    | CNN + Transformer + CTC        | Hybrid       | 7.7M           | **128 × 1024** |
| **VLT**        | Barrere et al., 2024       | `v_light_barrere.yaml`   | CNN + Transformer + CTC        | Hybrid       | 5.6M           | **128 × 1024** |



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

### ✍🏻 Real data

Links for downloading the datasets and splits used in the paper:
* IAM: Images [lines](https://fki.tic.heia-fr.ch/DBs/iamDB/data/lines.tgz), [lines GT](https://fki.tic.heia-fr.ch/DBs/iamDB/data/xml.tgz) and [Aachen split](https://www.openslr.org/resources/56/splits.zip).
* Rimes: [Complete](https://storage.teklia.com/public/rimes2011/RIMES-2011-Lines.zip). Link in: https://teklia.com/research/rimes-database/
* Bentham: https://zenodo.org/records/44519. [Complete](https://zenodo.org/records/44519/files/BenthamDatasetR0-GT.tbz?download=1)
* Saint Gall: [Complete](https://fki.tic.heia-fr.ch/DBs/iamHistDB/data/saintgalldb-v1.0.zip)
* George Washington: [Complete](https://fki.tic.heia-fr.ch/DBs/iamHistDB/data/washingtondb-v1.0.zip)
* Rodrigo: [Complete](https://zenodo.org/records/1490009/files/Rodrigo%20corpus%201.0.0.tar.gz?download=1)
* ICFHR2016: [Train and validation](https://zenodo.org/records/1164045/files/Train-And-Val-ICFHR-2016.tgz?download=1) [Test](https://zenodo.org/records/1164045/files/Test-ICFHR-2016.tgz?download=1)
<!-- $``$ -->

### 🤖 Synthetic data
We downloaded the data from [1001fonts](https://www.1001fonts.com/handwritten-fonts.html) and we manually filtered them. We include the complete names of the synthetic fonts (about 3600 fonts) used in the following [file](TODO.com)


#### 📚 Dataset Licensing and Attribution

The dataset used in this project is a **derived and preprocessed subset** of the [Wikipedia-based Image Text (WIT)](https://github.com/google-research-datasets/wit) dataset, originally released by Google Research.

The original WIT dataset is made available under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license.

#### 🔄 Modifications made:
- We selected an equal number of text-only samples across **five languages (en, fr, es, de, la)** to create a balanced multilingual subset.
- This number is given by the Latin (la) dataset which contain the smallest number of samples. 

This derived dataset is intended **solely for research purposes**. Please refer to the original [WIT GitHub repository](https://github.com/google-research-datasets/wit) for full license details and attribution requirements.

If you use this subset, we kindly ask you to cite the original WIT paper and repository. 


### Data preparation:

1. Move all the files to the data/ folder.
2. In data/ we should get the following (12) list of files executing: ```ls -1 data```:
```bash
BenthamDatasetR0-GT.tbz
RIMES-2011-Lines.zip
'Rodrigo corpus 1.0.0.tar'
Test-ICFHR-2016.tar
Train-And-Val-ICFHR-2016.tar
lines.tar
saintgalldb-v1.0.zip
splits.zip
synth-data.zip
vocab.txt
washingtondb-v1-3.0.zip
xml.tar
```

3. To preprocess and organize the datasets, use:
```bash
bash scripts/prepare_data.sh data/
```

4. We should get something similar to this, executing:
```bash
tree -dL 3 data
```
```plaintext
data/
├── htr_datasets
│   ├── bentham
│   ├── iam
│   ├── icfhr_2016
│   ├── rimes
│   ├── rodrigo
│   ├── saint_gall
│   └── washington
└── synth
    ├── fonts
    └── wit_dataset
```

❗️Warning: To use the synthetic fonts, you must include the .ttf files in the fonts/ folder.

---

## 🏗️ Training

To train an HTR model from scratch:

```bash
python src/train_ctc.py \
paths.data_dir='data/' \
data/train/train_config/datasets=[iam] \
data.train.train_config.img_size=[128,1024] \
data.train.train_config.batch_size=16 \
data.train.train_config.binarize=True \
data.train.train_config.num_workers=8 \
trainer.max_epochs=10 \
trainer.deterministic=False \
model=crnn_puig \
tokenizer=tokenizers/char_tokenizer \
callbacks.early_stopping.patience=100 \
callbacks.model_checkpoint_base.filename=crnn_puig_src_iam_check \
callbacks/heldout_targets=[rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
callbacks/optim_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016] \
logger.wandb.offline=False \
logger.wandb.name=crnn_puig_src_iam_check \
train=True 
```

This will start a run and start a project called "HTR-OOD" if you enter your Wandb key. 

### 📜 Explanation of Parameters

| Parameter | Description |
|-----------|-------------|
| `data/train/train_config/datasets=[iam]` | Specifies the dataset used for training (`IAM` in this case). |
| `data/val/val_config/datasets=[iam,rimes,...]` | Specifies the dataset used for validation (`IAM` in this case). If not specified, by default all datasets will be used for validation. 
| `data/test/test_config/datasets=[iam,rimes,...]` | Specifies the dataset used for testing (`IAM` in this case). If not specified, by default all datasets will be used for testing. 
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
| `callbacks/heldout_targets=[rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016]` | Specifies datasets **not seen during training** for out-of-distribution (OOD) evaluation. This will create the N-1 checkpoints (with suffix tgt_{target}) optimized using a leave-one-out for later testing on the excluded. |
| `callbacks/optim_targets=[iam,rimes,washington,saint_gall,bentham,rodrigo,icfhr_2016]` | Lists datasets used for **optimization and tuning**. This will create N checkpoints (with suffix optim_{target}), each one optimized for the target dataset. |
| `logger.wandb.offline=False` | Enables **online** tracking using **Weights & Biases (WandB)**. |
| `logger.wandb.name=crnn_puig_src_iam` | Sets the WandB experiment name to **crnn_puig_src_iam**. |
| `train=True` | ** Sets the model in training mode 
Notes: all the checkpoints will be created by default in a folder checkpoints/
---

## 🧪 Evaluation (from pretrained checkpoint)

To evaluate a pretrained model:

```bash
python src/train_ctc.py \
data/train/train_config/datasets=[iam] \
data.train.train_config.img_size=[128,1024] \
data.train.train_config.batch_size=8 \
data.train.train_config.binarize=True \
data.train.train_config.num_workers=8 \
trainer.max_epochs=500 \
trainer.deterministic=False \
model=crnn_puig \
tokenizer=tokenizers/char_tokenizer \
logger.wandb.offline=False \
logger.wandb.name=crnn_puig_src_iam_test \
train=False \
+pretrained_checkpoint=crnn_puig_src_iam_check_ID
```

---

<!-- ## 📈 Results

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

--- -->

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
This research was supported by the Spanish Ministry of Science and Innovation through the LEMUR research project (PID2023-148259NB-I00), funded by MCIU/AEI/10.13039/501100011033/FEDER, EU, and the European Social Fund Plus (FSE+). The first author is supported by grant CIACIF/2021/465 from “Programa I+D+i de la Generalitat Valenciana”. This research was supported by the University of Alicante.

---

## 📝 License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## 📬 Contact

For any questions, feel free to **open an issue** or reach out via carlos.garrido@ua.es.

---
## 😊 Contribute
✨ *Star this repository or share it if you find it helpful!* ⭐

