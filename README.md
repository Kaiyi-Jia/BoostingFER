# Sota-weight-and-val-code
Sota-weight and val code of "Boosting Representation Learning for High-Level Semantic Information in Facial Expression Recognition"

## 📦 权重文件

预训练权重通过 [GitHub Releases](../../releases) 提供，请从 **Latest Release** 下载所需文件。

| 文件名 | 任务 | 训练数据集 |
|--------|------|------------|
| `sota_ferplus.pth` | 7 分类 | FERPlus |
| `sota_raf-db.pth` | 7 分类 | RAF‑DB |
| `sota_affect7.pth` | 7 分类 | AffectNet (7 类) |
| `sota_affect8.pth` | 8 分类 | AffectNet (8 类) |

---

## 🧠 情绪类别

### 7 分类
`0: Neutral` `1: Happy` `2: Sad` `3: Surprised`  
`4: Scared` `5: Disgusted` `6: Angry`

### 8 分类（在 7 类基础上增加）
`7: Contemptuous`

---

## ⚙️ 环境要求

- Python 3.8+
- PyTorch ≥ 1.10（推荐 CUDA 11.7+）
- OpenAI CLIP（官方库）
- torchvision, tqdm, numpy

---

## 🚀 快速开始

### 1. 下载权重
从 [Releases](../../releases) 页面下载所需的 `.pth` 文件，放入项目根目录。

### 2. 准备测试数据
测试集根目录下需包含按类别编号命名的子文件夹，例如：
