# 🚀 Event-Driven Intrusion Detection System using Spiking Neural Networks (SNN-IDS)

A **high-fidelity, event-driven Intrusion Detection System (IDS)** designed for IoT and edge environments, leveraging **Spiking Neural Networks (SNNs)** and **neuromorphic computing principles** to achieve **energy-efficient and real-time cyber threat detection**.

📄 Based on IEEE research:  
*Event-Driven Intrusion Detection Systems using Spiking Neural Networks for Edge and IoT Security*

---

## 📋 Table of Contents

1. [Overview](#1--overview)  
2. [Problem Statement](#2--problem-statement)  
3. [Proposed System Architecture](#3--proposed-system-architecture)  
4. [End-to-End Workflow](#4--end-to-end-workflow)  
5. [Datasets](#5--datasets)  
6. [Data Preprocessing](#6--data-preprocessing)  
7. [Spike Encoding (TTFS)](#7--spike-encoding-ttfs)  
8. [SNN Model Architecture](#8--snn-model-architecture)  
9. [Learning Mechanism (STDP)](#9--learning-mechanism-stdp)  
10. [Training Process](#10--training-process)  
11. [Evaluation & Metrics](#11--evaluation--metrics)  
12. [Results & Analysis](#12--results--analysis)  
13. [Comparison with Baselines](#13--comparison-with-baselines)  
14. [Code Architecture](#14--code-architecture)  
15. [Setup & Usage](#15--setup--usage)  
16. [Key Insights](#16--key-insights)  
17. [Limitations](#17--limitations)  
18. [Future Work](#18--future-work)  
19. [Team](#19--team)  

---

## 1. 🔭 Overview

With the rapid growth of IoT devices, network environments have become:

- Highly distributed  
- Resource-constrained  
- Vulnerable to evolving cyber attacks  

Traditional IDS systems fail to meet real-time and energy constraints.

### ✅ This project introduces:

- Event-driven intrusion detection using spikes  
- Temporal pattern learning using SNNs  
- Ultra-sparse computation for efficiency  
- A full pipeline from raw data → detection  

---

## 2. 🔍 Problem Statement

> How can we detect network intrusions efficiently on edge devices without sacrificing accuracy?

### Challenges:

- High-dimensional network traffic  
- Real-time processing requirements  
- Energy constraints in IoT devices  
- Temporal nature of attack patterns  

### Limitations of Traditional Methods:

- Signature-based IDS → cannot detect unknown attacks  
- Deep learning → high computational cost  
- Static models → poor temporal understanding  

---

## 3. 🏗️ Proposed System Architecture

| Module | Function | Output |
|--------|----------|--------|
| Data Processing | Cleans and selects features | Normalized features |
| Spike Encoder | Converts features → spike trains | Sparse spike matrix |
| SNN Model | Learns temporal patterns | Attack classification |

---

## 4. ⚡ End-to-End Workflow


Raw Network Traffic
↓
Data Cleaning (NaN, Inf removal)
↓
Feature Selection (Top 16–20)
↓
Normalization (MinMax)
↓
Spike Encoding (TTFS)
↓
SNN Processing (LIF neurons)
↓
STDP Learning
↓
Spike Count Classification
↓
Final Attack Prediction


---

## 5. 📊 Datasets

| Dataset | Samples | Features | Description |
|--------|--------|----------|------------|
| TON_IoT | ~2.3M | 44 → 16 | IoT telemetry attacks |
| CIC-IDS2018 | ~3.1M | 78 → 20 | Enterprise network attacks |

### Final Classes:


Normal | DoS | DDoS | Scan | Others


---

## 6. 🧹 Data Preprocessing

### Steps:

1. Remove missing values and duplicates  
2. Replace Inf/-Inf values  
3. Label mapping → 5 classes  
4. Feature selection:
   - Mutual Information (MI)
   - Recursive Feature Elimination (RFE)  
5. Normalize features:


x' = (x - min) / (max - min)


---

## 7. ⚡ Spike Encoding (TTFS)

### Time-To-First-Spike


t_spike = feature_value × 100 ms


### Encoding Rules:

- Spike only if value ≥ threshold  
- Threshold = mean of feature  
- Refractory period = 5 ms  

### Sparsity:

| Dataset | Sparsity |
|--------|---------|
| TON_IoT | 99.9% |
| CIC | 99.7% |

---

## 8. 🧠 SNN Model Architecture

| Layer | Neurons | Description |
|------|--------|------------|
| Input | 16–20 | Spike input neurons |
| Hidden | 40 | LIF neurons |
| Output | 5 | One per class |

### LIF Equation:


τ dV/dt = -V + Σ(w × spikes)


---

## 9. 🔁 Learning Mechanism (STDP)


Δw = A+ e^(-Δt/τ+) if pre before post
Δw = -A- e^(Δt/τ-) if post before pre


---

## 10. 🏋️ Training Process

- Unsupervised learning  
- 30 epochs  
- 100 ms simulation per sample  

---

## 11. 📏 Evaluation & Metrics

- Accuracy  
- F1 Score  
- Latency (ms/sample)  
- Energy (nJ/sample)  

---

## 12. 📈 Results & Analysis

### TON_IoT

| Metric | Value |
|-------|------|
| Accuracy | 98.5% |
| F1 Score | 0.985 |
| Latency | 159.8 ms |
| Energy | 1.370 nJ |

### CIC-IDS2018

| Metric | Value |
|-------|------|
| Accuracy | 77.0% |
| F1 Score | 0.757 |
| Latency | 124.8 ms |
| Energy | 1.927 nJ |

---

## 13. 📊 Comparison with Baselines

### TON_IoT

| Model | Accuracy | F1 |
|------|---------|----|
| SVM | 85.5% | 0.85 |
| KNN | 96.6% | 0.96 |
| Random Forest | 97.85% | 0.978 |
| **SNN (Ours)** | **98.5%** | **0.985** |

### CIC-IDS2018

| Model | Accuracy | F1 |
|------|---------|----|
| SVM | 81.7% | 0.81 |
| KNN | 81.4% | 0.81 |
| Random Forest | 81.9% | 0.817 |
| **SNN (Ours)** | **77.0%** | **0.757** |

---

## 14. 🗂️ Code Architecture


src/
├── preprocess.py
├── preprocess_cic.py
├── spike_encoder.py
├── snn_model.py
├── train_stdp.py
├── evaluate.py
├── test_encoder.py
├── test_snn.py


---

## 15. 🚀 Setup & Usage

### Installation

```bash
python -m venv snn_env
source snn_env/bin/activate
pip install brian2 numpy pandas scikit-learn matplotlib
Run Pipeline
python src/preprocess.py
python src/preprocess_cic.py
python src/train_stdp.py
python src/evaluate.py

16. 💡 Key Insights

SNN captures temporal attack behavior
Event-driven design reduces energy usage
Strong performance on structured IoT data
CIC dataset highlights generalization challenges

17. ⚠️ Limitations

Issue	Reason
Lower CIC accuracy	Complex traffic
Higher latency	Simulation overhead
Unsupervised learning	Needs post-label mapping

18. 🔮 Future Work

Hybrid SNN + Deep Learning
Improved spike encoding
Neuromorphic hardware deployment
Adaptive thresholds

19. 👨‍💻 Team
Rakesh
Ujwal
Prajwal
Tarun
