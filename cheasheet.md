# 📚 Cheatsheet UTS Machine Learning
## Multiple Feature, Overfitting, dan Perbaikannya

---

## 🗂️ Daftar Isi
1. [Alur Kerja Machine Learning](#1-alur-kerja-machine-learning)
2. [Persiapan Dataset](#2-persiapan-dataset)
3. [Fungsi-Fungsi Utama (Predefined Code)](#3-fungsi-fungsi-utama)
4. [Training Model & Evaluasi](#4-training-model--evaluasi)
5. [Multiple Feature](#5-multiple-feature)
6. [Overfitting & Underfitting](#6-overfitting--underfitting)
7. [Feature Selection](#7-feature-selection)
8. [Regularisasi (L2)](#8-regularisasi-l2)
9. [Efek Jumlah Data](#9-efek-jumlah-data)
10. [Ringkasan Konsep Kunci](#10-ringkasan-konsep-kunci)

---

## 1. Alur Kerja Machine Learning

```
Dataset → Scaling → Add Bias → Train/Test Split → Train/Val Split
    → Training (Gradient Descent) → Evaluasi (MSE)
```

---

## 2. Persiapan Dataset

### a. Baca Dataset
```python
import numpy as np, pandas as pd
data = pd.read_csv("url_atau_path_dataset")
data.info()         # lihat tipe data & shape
data.describe()     # statistik setiap fitur
```

### b. Pisahkan Fitur Input (X) dan Label (y)
```python
X = data.drop(columns=["harga_juta_rupiah"]).to_numpy()
y = data["harga_juta_rupiah"].to_numpy()
```

### c. Min-Max Scaling
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

```python
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)
# Hasil: semua fitur dalam range [0, 1]
```

> **Kenapa scaling?**  
> Menyamakan skala antar fitur → konvergensi gradient descent lebih **cepat**.  
> Tanpa scaling → proses training sangat **lambat**.

### d. Tambahkan Bias
```python
X = add_bias(X)   # menambahkan kolom 1 di depan X (intercept / w0)
# Shape: (n_samples, n_features) → (n_samples, n_features+1)
```

### e. Split Train / Test (80:20)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
# X_train: 1200 data | X_test: 300 data (dari total 1500)
```

> **Kenapa split?**  
> Mencegah **overfitting** (model tidak hanya "hafal" data training).

### f. Split Train / Validation (dari data training)
```python
X_train_sungguhan, X_val, y_train_sungguhan, y_val = train_test_split(X_train, y_train, 0.2)
```

---

## 3. Fungsi-Fungsi Utama

```python
def add_bias(x):
  bias = np.ones(x.shape[0])
  return np.c_[bias, x]

def cost(y,pred, w=None,lamda=0.0):
    m = y.shape[0]
    mse = ((pred - y) ** 2).sum() / (2 * m)
    reg = 0
    if w is not None and lamda!=0.0:
      reg = (lamda / (2 * m)) * (w[1:] ** 2).sum()   # w0 tidak ikut
    return mse + reg

def predict(w,x):
  return x @ w

def update_bobot(w,xb,y,alpha, lamda=0.0):
  output = predict(w,xb)
  error = output-y
  m = y.shape[0]
  #rergularization
  reg = (lamda / m) * w
  reg[0] = 0
  gradient = (xb.T @ error) / m
  w = w - alpha*(gradient+reg)
  return w

def train(X, y, X_val, y_val, alpha=0.1, iters=500, verbose=True, lamda=0.0):
    np.random.seed(RANDSEED)
    w = np.random.randn(X.shape[1]) * 0.01
    history_train = []
    history_val = []

    for i in range(iters):
        pred = predict(w, X)
        loss = cost(y, pred, w, lamda)
        history_train.append(loss)

        pred_val = predict(w, X_val)
        loss_val = cost(y_val, pred_val)
        history_val.append(loss_val)
        if verbose:
          print("Iterasi ke-",i,"  Train Loss: ",loss, "  Val Loss:",loss_val)
        w = update_bobot(w, X, y, alpha, lamda)

    return w, np.array(history_train), np.array(history_val)

def train_test_split(X, y, test_ratio=0.2, seed=RANDSEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def pilih_sample_random(X, y, n, seed=RANDSEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], n, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]
    return X_sample, y_sample

def plot_history_train_validation(hist_train, hist_val):
    plt.plot(hist_train, label="Train Loss")
    plt.plot(hist_val, label="Validation Loss")
    plt.xlabel("Iterasi")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.show()

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
```

---

## 4. Training Model & Evaluasi

```python
# Training
w, hist_train, hist_val = train(X_train_sungguhan, y_train_sungguhan,
                                X_val, y_val, alpha=0.1, iters=2000)

# Evaluasi MSE
error_train = mse(y_train_sungguhan, predict(w, X_train_sungguhan))  # Memorization
error_test  = mse(y_test,            predict(w, X_test))             # Generalization
```

### Eksperimen Hyperparameter (Learning Rate & Iterasi)

| Eksperimen | Iterasi | Learning Rate | MSE Train | MSE Test |
|:----------:|:-------:|:-------------:|:---------:|:--------:|
| 1 | 500 | 0.5 | 315.11 | 321.12 |
| 2 | 2000 | 0.5 | 313.50 | 319.14 |
| 3 | 1000 | 0.1 | 360.73 | 371.51 |
| **4** ✅ | **2000** | **0.1** | **317.80** | **324.24** |
| 5 | 2000 | 0.01 | 2479.79 | 2593.96 |

> **Terbaik:** `alpha=0.1`, `iters=2000`

---

## 5. Multiple Feature

### Eksperimen Kombinasi Fitur

| Fitur yang Digunakan | MSE Train | MSE Test |
|:--------------------:|:---------:|:--------:|
| **Semua (1–12)** ✅ | **726.99** | **757.57** |
| Fitur 1 saja | 11835.55 | 13855.29 |
| Fitur 2 saja | 60570.09 | 57031.43 |
| Fitur 3 saja | 43141.22 | 43912.28 |
| Fitur 1, 2 | 4205.48 | 4512.13 |
| Fitur 1, 3 | 9691.58 | 11747.16 |
| Fitur 2, 3 | 26985.36 | 24990.04 |
| Fitur 1, 2, 3 | 3916.81 | 4270.85 |

> **Kesimpulan:** Semakin banyak fitur relevan → error **semakin kecil**.  
> Multiple feature memberikan **informasi lebih lengkap** sehingga model lebih akurat.

---

## 6. Overfitting & Underfitting

### Skenario Eksperimen (data kecil ~12 sampel, `X_train_small`)

| Skenario | Fitur | MSE Train | MSE Test | Kondisi |
|:--------:|:-----:|:---------:|:--------:|:-------:|
| **b** | 1 fitur (0,1) | 8427.37 | 4398.03 | 🔵 **Underfitting** |
| **c** ✅ | 4 fitur (0–4) | 257.82 | 735.75 | ✅ **Ideal** |
| **d** | Semua fitur | 156.91 | 2108.68 | 🔴 **Overfitting** |

### Ciri-Ciri di Grafik Training Curve

| Kondisi | Train Loss | Val Loss | Penyebab |
|:-------:|:----------:|:--------:|:--------:|
| **Underfitting** | Tinggi & stagnan | Tinggi & stagnan | Fitur terlalu sedikit, iterasi kurang, model terlalu sederhana |
| **Overfitting** | Terus turun → 0 | Naik / tetap tinggi | Data terlalu sedikit, fitur terlalu banyak, iterasi terlalu banyak |
| **Ideal** | Turun & konvergen | Turun & konvergen berdekatan dengan train | Fitur relevan seimbang, data cukup |

```
Underfitting            Ideal              Overfitting
Loss ↑                 Loss ↑             Loss ↑
  ─────── train          ╲ train            ╲ train
  ─────── val             ╲___val            ╲____
                                              ╱ val (naik)
         Iterasi →        Iterasi →          Iterasi →
```

---

## 7. Feature Selection

### Sequential Forward Selection (SFS)
- Mulai dari **0 fitur**, tambahkan 1 fitur terbaik tiap langkah.
- **Terbaik:** Fitur `[0, 1, 2, 3, 4, 5]` → MSE Test = **689.29**

### Sequential Backward Elimination (SBFE)
- Mulai dari **semua fitur**, hapus 1 fitur terlemah tiap langkah.
- **Terbaik:** Fitur `[0, 1, 2, 3, 4, 5, 6, 7, 8, 12]` → MSE Test = **660.86**

```python
# Contoh memilih subset fitur
idx_fitur = [0, 1, 2, 3, 4, 5]   # kolom 0 = bias, selalu disertakan
X_fitur   = X_train_small[:, idx_fitur]
X_val_f   = X_val_small[:, idx_fitur]
w, ht, hv = train(X_fitur, y_train_small, X_val_f, y_val_small, 0.01, 10000, False)
```

---

## 8. Regularisasi (L2)

### Formula Cost Function + Regularisasi
$$J(w) = \frac{1}{2m}\sum(ŷ - y)^2 + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2$$

> `w0` (bias) **tidak** ikut diregularisasi.

### Pengaruh Lambda (λ) — semua fitur, data kecil

| λ | MSE Train | MSE Test | Kondisi |
|:-:|:---------:|:--------:|:-------:|
| 0.0001 | 157.03 | 2108.54 | 🔴 Overfitting |
| 0.001 | 158.09 | 2107.32 | 🔴 Overfitting |
| **0.01** ✅ | **170.56** | **2097.85** | ✅ Terbaik |
| 0.1 | 428.64 | 2184.43 | ⚠️ Mulai Underfitting |
| 1 | 4602.95 | 4180.44 | 🔵 Underfitting |
| 10 | 16322.91 | 8164.96 | 🔵 Underfitting |
| 100 | 21265.70 | 9915.15 | 🔵 Underfitting |

> **Kesimpulan:**
> - λ terlalu kecil → **Overfitting** (regularisasi tidak cukup kuat)
> - λ terlalu besar → **Underfitting** (bobot ditekan terlalu keras)
> - **Sweet spot: λ = 0.01**

```python
lamda = 0.01
w, ht, hv = train(X_train_small, y_train_small, X_val_small, y_val_small,
                  alpha=0.01, iters=10000, verbose=False, lamda=lamda)
```

---

## 9. Efek Jumlah Data

```python
rasio_list = [0.1, 0.2, 0.3, ..., 1.0]
for rasio in rasio_list:
    n = int(rasio * X_train_sungguhan.shape[0])
    X_r, y_r = pilih_sample_random(X_train_sungguhan, y_train_sungguhan, n)
    w, _, _ = train(X_r, y_r, X_val, y_val, 0.01, 10000, False)
    # hitung MSE train & test ...
```

**Kesimpulan:**  
Semakin banyak data training → MSE Train & Test **turun** dan **merapat** → model lebih **stabil** dan terhindar dari overfitting.

```
MSE ↑
  ╲                    ← Train MSE
   ╲_______________
  ╲                    ← Test MSE
   ╲____
        ───────────── Jumlah Data →
```

---

## 10. Ringkasan Konsep Kunci

| Konsep | Definisi Singkat | Solusi jika Bermasalah |
|:------:|:----------------:|:----------------------:|
| **Min-Max Scaling** | Normalisasi fitur ke [0,1] | Wajib sebelum training |
| **Bias (w0)** | Intercept model, kolom 1 di X | Tambahkan dengan `add_bias()` |
| **Train/Val/Test Split** | Pisahkan data untuk training, tuning, evaluasi akhir | Rasio umum: 60/20/20 atau 80/10/10 |
| **MSE** | Mean Squared Error = metrik loss | Semakin kecil semakin baik |
| **Memorization Error** | MSE pada data training | — |
| **Generalization Error** | MSE pada data testing | Harus mendekati train error |
| **Learning Rate (α)** | Ukuran langkah update bobot | Terlalu besar: divergen; Terlalu kecil: lambat |
| **Underfitting** | Model terlalu sederhana | Tambah fitur, tambah iterasi |
| **Overfitting** | Model terlalu hafal data training | Feature selection, regularisasi, tambah data |
| **Feature Selection (SFS/SBFE)** | Pilih subset fitur terbaik | Gunakan validasi sebagai panduan |
| **Regularisasi L2 (Ridge)** | Penalti besar bobot via λ | Tuning λ dengan validasi |
| **Penambahan Data** | Lebih banyak data → generalisasi lebih baik | Kumpulkan / augmentasi data |

---

### 🔑 Formula Penting

| Formula | Keterangan |
|:-------:|:----------:|
| $ŷ = Xw$ | Prediksi linear |
| $J = \frac{1}{2m}\sum(ŷ-y)^2$ | MSE Loss |
| $J_{reg} = J + \frac{\lambda}{2m}\sum w_j^2$ | MSE + L2 Regularisasi |
| $w := w - \alpha \nabla J$ | Update Gradient Descent |
| $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Min-Max Scaling |

---

*Cheatsheet ini dibuat dari praktikum Matkul Machine Learning — UTS*