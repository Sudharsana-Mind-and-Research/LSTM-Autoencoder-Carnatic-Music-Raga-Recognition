# 🎵 Carnatic Raga Recognition using LSTM-Autoencoder

> **Published Research** — Rajalaxmi R.R., Sudharsana P.P., Thangarajan R. (2026). *An Effective LSTM–Autoencoder Approach with Acoustic Features in Indian Classical Music Raga Recognition.* SPELLL 2024, Communications in Computer and Information Science, vol 2656, Springer, Cham.
> 🔗 [https://doi.org/10.1007/978-3-032-05855-3_3](https://doi.org/10.1007/978-3-032-05855-3_3)

---

## 📌 Overview

This project addresses the automatic identification of ragas in **South Indian Carnatic Music** using deep learning. Ragas are the fundamental melodic frameworks of Indian Classical Music, each carrying a distinct emotional and aesthetic character. Automating raga recognition is a challenging problem in **Music Information Retrieval (MIR)** due to the intricate improvisational grammar and tonal subtleties involved.

We curate a custom dataset of **600 Carnatic music audio clips** spanning **20 ragas**, extract two sets of acoustic features (spectral and rhythmic), and benchmark five deep learning models — with the **LSTM-Autoencoder achieving 94.17% accuracy**.

---

## 🎼 Ragas Covered (20 Ragas)

| | | | |
|---|---|---|---|
| Anandabhairavi | Atana | Bageda | Bhairavi |
| Bilahari | Dhanyasi | Hamsadhwani | Harikambodi |
| Hindolam | Kalyani | Kambodhi | Kharaharapriya |
| Madhyamavathi | Mayamalavagowla | Mohanam | Mukhari |
| Natakurunji | Panthuvarali | Purvikalyani | Reetigowla |

---

## 📁 Repository Structure

```
├── datasets/
│   ├── raga_recog_dataset.csv              # Dataset 1 — Feature Set 1 (26 features, 260 songs)
│   └── raga_dataset_RythmFeatrues_phase-II.csv  # Dataset 2 — Feature Set 2 (51 features, 600 songs)
│
├── notebooks/
│   ├── feature_feb_20.ipynb                # Feature extraction pipeline (librosa-based)
│   └── Raga_LSTM_mar_9__1_.ipynb           # Deep learning model training & evaluation
│
├── paper/
│   └── 23_Manuscript.pdf                   # Published manuscript (SPELLL 2024, Springer)
│
└── README.md
```

---

## 🔬 Feature Extraction

Features are extracted from audio files using **librosa** with a fixed duration of 40 seconds per clip.

### Feature Set 1 — Spectral Features (26-dim vector)

| Feature | Description |
|--------|-------------|
| Zero Crossing Rate (ZCR) | Rate of sign changes in the audio signal |
| MFCCs (×20) | Mel-Frequency Cepstral Coefficients — timbral texture |
| Spectral Centroid | Weighted mean / "centre of mass" of frequencies |
| Spectral Rolloff | Frequency below which a set percentage of total energy lies |
| Chroma Frequencies | 12-bin projection of the spectrum onto musical semitones |
| Root Mean Square Energy | Signal loudness / energy |

### Feature Set 2 — Spectral + Rhythmic Features (51-dim vector)

Extends Feature Set 1 with:

| Feature | Description |
|--------|-------------|
| Tonnetz | Harmonic relationships in tonal space |
| Spectral Contrast | Peak-valley difference across frequency sub-bands |
| Mel Spectrogram | Frequency spectrum mapped to mel scale |
| Tempogram | Autocorrelation of onset strength — rhythmic periodicity |
| Onset | Amplitude rise detection — timing and rhythmic structure |
| MFCCs (×40) | Extended MFCC range for better timbral resolution |

---

## 🧠 Models & Results

Five deep learning models are evaluated across three experiment configurations:

| Model | Exp 1 (D1-FS1) | Exp 2 (D1-FS2) | Exp 3 (D2-FS2) |
|-------|:--------------:|:--------------:|:--------------:|
| ANN | 19.00% | 23.08% | 63.33% |
| CNN | 32.31% | 50.77% | 71.33% |
| LSTM | 21.54% | 23.08% | 92.67% |
| CNN-LSTM | 66.15% | 69.23% | 88.97% |
| **LSTM-Autoencoder** | **70.77%** | **75.38%** | **94.17% ✅** |

> **D1** = Dataset 1 (260 songs), **D2** = Dataset 2 (600 songs), **FS1/FS2** = Feature Set 1/2

### LSTM-Autoencoder Architecture

```
Input (1, 51)
    → LSTM(128, ReLU) → LSTM(64, ReLU, dropout=0.05)   [Encoder]
    → RepeatVector(timesteps)
    → LSTM(64, ReLU) → LSTM(128, ReLU)                  [Decoder]
    → TimeDistributed(Flatten)
    → Dense(20, Softmax)                                 [Classification]
```

- **Optimizer**: Adam (lr = 0.0001)
- **Loss**: Sparse Categorical Cross-Entropy
- **Epochs**: 200 | **Batch size**: 10

---

## 📊 Dataset

- **Phase I**: 260 audio clips — 13 songs × 20 ragas (mp3 format)
- **Phase II**: 600 audio clips — 30 songs × 20 ragas (mp3 format)
- Primary source: [Raga Surabhi](https://www.ragasurabhi.com/carnatic-music/ragas.html)
- Dataset curated and annotated by the authors

---

## ⚙️ Requirements

```bash
pip install librosa numpy pandas scikit-learn tensorflow keras jupyter
```

---

## 🚀 Usage

### 1. Feature Extraction

Open and run `notebooks/feature_feb_20.ipynb` with your audio directory. The notebook extracts features for each audio file and saves them to CSV.

### 2. Model Training & Evaluation

Open `notebooks/Raga_LSTM_mar_9__1_.ipynb` to train and evaluate all five deep learning models on the extracted features.

---

## 📎 Citation

If you use this dataset or code in your work, please cite:

```bibtex
@inproceedings{rajalaxmi2026lstm,
  author    = {Rajalaxmi, R.R. and Sudharsana, P.P. and Thangarajan, R.},
  title     = {An Effective {LSTM}--Autoencoder Approach with Acoustic Features in Indian Classical Music Raga Recognition},
  booktitle = {Speech and Language Technologies for Low-Resource Languages. SPELLL 2024},
  series    = {Communications in Computer and Information Science},
  volume    = {2656},
  publisher = {Springer, Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-05855-3_3}
}
```

---

## 👩‍🔬 Authors

- **Rajalaxmi R.R.** — Department of CSE, Kongu Engineering College, Perundurai
- **Sudharsana P.P.** — Department of CSE, Kongu Engineering College, Perundurai
- **Thangarajan R.** — Department of CSD, Kongu Engineering College, Perundurai

---

## 🔮 Future Work

- Incorporating Nature-Inspired Optimization (Particle Swarm, Cat Swarm) for hyperparameter tuning
- Extending to **Hindustani Classical Music** raga recognition
- Real-time raga identification from live audio streams
