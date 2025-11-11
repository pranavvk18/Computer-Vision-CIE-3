
# Image Deblurring using U-Net (PyTorch)

This project implements **image deblurring** using a **U-Net convolutional neural network**.  
It aims to restore sharp images from motion- or defocus-blurred inputs using supervised learning.

---

## 📁 Dataset Structure

Dataset used: **GoPro Deblurring Dataset (subset)** or custom blurred images.

```
gopro_deblur/
├── blur/
│   └── images/
│       ├── 000001.png
│       ├── 000002.png
│       └── ...
└── sharp/
    └── images/
        ├── 000001.png
        ├── 000002.png
        └── ...
```

---

## ⚙️ Setup Instructions

### 1️⃣ Run on Google Colab

- Upload your dataset to Google Drive under the given folder structure.
- Open the Colab notebook.
- Enable GPU runtime:  
  `Runtime > Change runtime type > Hardware accelerator > GPU`.

### 2️⃣ Install Dependencies

```bash
!pip install torch torchvision scikit-image tqdm opencv-python
```

### 3️⃣ Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧩 Model Architecture

Based on U-Net encoder–decoder with skip connections.

Captures both global blur patterns and fine texture details.

Trained with L1 Loss using Adam optimizer.

---

## 🚀 Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Model | U-Net |
| Loss Function | L1 (Mean Absolute Error) |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 20 |
| Batch Size | 4 |
| Image Size | 256×256 |
| Training Samples | 150 pairs (randomly selected) |

---

## 📊 Evaluation Metrics

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

```python
psnr = peak_signal_noise_ratio(sharp_np, pred_np, data_range=1.0)
ssim = structural_similarity(sharp_np, pred_np, channel_axis=2, data_range=1.0)
```