# Sota-weight-and-val-code
Sota-weight and val code of "Boosting Representation Learning for High-Level Semantic Information in Facial Expression Recognition"

## 📦 Weight Files

Pretrained weights are available via [GitHub Releases](../../releases) ，Please download the required files from the **Latest Release**.

| Filename | Task | Training Dataset |
|--------|------|------------|
| `sota_ferplus.pth` | 7-class classification | FERPlus |
| `sota_raf-db.pth` | 7-class classification | RAF‑DB |
| `sota_affect7.pth` | 7-class classification | AffectNet (7 classes) |
| `sota_affect8.pth` | 8-class classification | AffectNet (8 classes) |

---

## 🧠 Emotion Classes

### 7-class Classification
`0: Neutral` `1: Happy` `2: Sad` `3: Surprised`  
`4: Scared` `5: Disgusted` `6: Angry`

### 8-class Classification (extends 7-class)
`7: Contemptuous`

---

## ⚙️ Environment Requirements

- Python 3.8+
- PyTorch ≥ 1.10（CUDA 11.7+ recommended）
- OpenAI CLIP（official library）
- torchvision, tqdm, numpy

---

## 🚀 Quick Start

### 1. Download Weights
Download the required `.pth` files from the [Releases](../../releases) and place them in the project root directory.

### 2. Prepare Test Data
The test set root directory must contain subfolders named by class numbers. For example: D:/val/
├── 0/
├── 1/
├── ...
└── 6/ (for 7-class) or 7/ (for 8-class)
### 3. Run Evaluation

#### Test 7-class weights
python evaluate.py --weights sota_affect7.pth --test_root D:/val --num_classes 7
#### Test 8-class weights
python evaluate.py --weights sota_affect8.pth --test_root D:/val --num_classes 8
#### Export results to CSV 
python evaluate.py --weights sota_affect7.pth sota_affect8.pth \
    --test_root D:/val --num_classes 7 --output results.csv
#### Export results to CSV
| Argument | Type | Description | Default |
|------|------|------|--------|
| `--weights` | multiple paths | **Required**, one or more weight file paths | - |
| `--test_root` | path | **Required**, test set root directory | - |
| `--num_classes` | 7 or 8 | Number of classes | `8` |
| `--batch_size` | int | Batch size | `64` |
| `--num_workers` | int | Number of data loading workers | `0` |
| `--device` | cuda/cpu | Execution device | `cuda`(if available)|
| `--output` | path | Optional CSV result save path | `None` |

```
Testing weight: sota_affect7.pth
✅ Weights loaded successfully.
Valid test samples: 4000 (classes 0~6)
🔥 Starting evaluation...
🎯 Overall Accuracy (7-class): 67.85% (2714/4000)

==================================================
Per-Class Accuracy:
  [0] Neutral     : 72.50% (580/800)
  [1] Happy       : 85.33% (683/800)
  [2] Sad         : 61.25% (490/800)
  [3] Surprised   : 76.13% (609/800)
  [4] Scared      : 55.00% (440/800)
  [5] Disgusted   : 52.88% (423/800)
  [6] Angry       : 62.63% (501/800)
==================================================
```
## 📁 Project Structure
.
├── evaluate.py                # Main evaluation script
├── README.md
└── (Weight files to be downloaded from Releases)
