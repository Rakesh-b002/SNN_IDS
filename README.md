# 🚀 Event-Driven Intrusion Detection System using Spiking Neural Networks (SNN-IDS)

A high-fidelity, event-driven Intrusion Detection System (IDS) designed for IoT and edge environments, leveraging Spiking Neural Networks (SNNs) and neuromorphic computing principles to achieve energy-efficient and real-time cyber threat detection.

Based on IEEE research:
Event-Driven Intrusion Detection Systems using Spiking Neural Networks for Edge and IoT Security

------------------------------------------------------------

## 📋 Table of Contents

1. Overview
2. Problem Statement
3. Proposed System Architecture
4. End-to-End Workflow
5. Datasets
6. Data Preprocessing
7. Spike Encoding (TTFS)
8. SNN Model Architecture
9. Learning Mechanism (STDP)
10. Training Process
11. Evaluation & Metrics
12. Results & Analysis
13. Comparison with Baselines
14. Code Architecture
15. Setup & Usage
16. Key Insights
17. Limitations
18. Future Work
19. Team

------------------------------------------------------------

## 1. Overview

With the rapid growth of IoT devices, network environments have become:
- Highly distributed
- Resource-constrained
- Vulnerable to evolving cyber attacks

Traditional IDS systems fail to meet real-time and energy constraints.

This project introduces:
- Event-driven intrusion detection using spikes
- Temporal pattern learning using SNNs
- Ultra-sparse computation (~99% sparsity)
- Complete pipeline from raw data to detection

------------------------------------------------------------

## 2. Problem Statement

How can we detect network intrusions efficiently on edge devices without sacrificing accuracy?

Challenges:
- High-dimensional network traffic
- Real-time processing requirements
- Energy constraints in IoT devices
- Temporal nature of attack patterns

Limitations of traditional methods:
- Signature-based IDS cannot detect unknown attacks
- Deep learning models are computationally expensive
- Static models fail to capture temporal behavior

------------------------------------------------------------

## 3. Proposed System Architecture

Module               Function                             Output
------------------------------------------------------------
Data Processing      Clean + feature selection            Normalized features
Spike Encoder        Convert features to spikes           Sparse spike trains
SNN Model            Learn temporal patterns              Attack classification

------------------------------------------------------------

## 4. End-to-End Workflow

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

------------------------------------------------------------

## 5. Datasets

Dataset         Samples     Features        Description
------------------------------------------------------------
TON_IoT         ~2.3M       44 → 16         IoT telemetry attacks
CIC-IDS2018     ~3.1M       78 → 20         Enterprise network attacks

Classes:
Normal | DoS | DDoS | Scan | Others

------------------------------------------------------------

## 6. Data Preprocessing

Steps:
- Remove missing values and duplicates
- Replace Inf / -Inf values
- Map labels to 5 classes
- Feature selection using:
  - Mutual Information (MI)
  - Recursive Feature Elimination (RFE)
- Normalize data:

x' = (x - min) / (max - min)

------------------------------------------------------------

## 7. Spike Encoding (TTFS)

Time-To-First-Spike method:

t_spike = feature_value × 100 ms

Rules:
- Spike only if value ≥ threshold
- Threshold = mean of feature
- Refractory period = 5 ms

Sparsity:
- TON_IoT → 99.9%
- CIC → 99.7%

------------------------------------------------------------

## 8. SNN Model Architecture

Layer       Neurons     Description
--------------------------------------
Input       16–20       Spike inputs
Hidden      40          LIF neurons
Output      5           Class neurons

LIF Equation:
τ dV/dt = -V + Σ(w × spikes)

------------------------------------------------------------

## 9. Learning Mechanism (STDP)

Δw = A+ e^(-Δt/τ+)   if pre before post
Δw = -A- e^(Δt/τ-)  if post before pre

- Strengthens useful connections
- Weakens irrelevant ones
- Learns temporal relationships

------------------------------------------------------------

## 10. Training Process

- Unsupervised learning
- 30 epochs
- 100 ms simulation window per sample

Steps:
1. Encode spikes
2. Run simulation
3. Capture spike activity
4. Update weights using STDP

------------------------------------------------------------

## 11. Evaluation & Metrics

- Accuracy
- F1 Score
- Latency (ms/sample)
- Energy (nJ/sample)

------------------------------------------------------------

## 12. Results & Analysis

TON_IoT:
- Accuracy : 98.5%
- F1 Score : 0.985
- Latency  : 159.8 ms
- Energy   : 1.370 nJ

CIC-IDS2018:
- Accuracy : 77.0%
- F1 Score : 0.757
- Latency  : 124.8 ms
- Energy   : 1.927 nJ

------------------------------------------------------------

## 13. Comparison with Baselines

TON_IoT:
SVM             → 85.5%
KNN             → 96.6%
Random Forest   → 97.85%
SNN (Ours)      → 98.5%

CIC-IDS2018:
SVM             → 81.7%
KNN             → 81.4%
Random Forest   → 81.9%
SNN (Ours)      → 77.0%

------------------------------------------------------------

## 14. Code Architecture

src/
├── preprocess.py
├── preprocess_cic.py
├── spike_encoder.py
├── snn_model.py
├── train_stdp.py
├── evaluate.py
├── test_encoder.py
├── test_snn.py

------------------------------------------------------------

## 15. Setup & Usage

Requirements:
- Python 3.9+
- 4 GB RAM

Installation:

python -m venv snn_env
source snn_env/bin/activate   (Linux/macOS)
snn_env\Scripts\activate      (Windows)

pip install brian2 numpy pandas scikit-learn matplotlib

Run:

python src/preprocess.py
python src/preprocess_cic.py
python src/train_stdp.py
python src/evaluate.py

------------------------------------------------------------

## 16. 💡 Key Insights

- **Temporal learning advantage**  
  SNN effectively captures time-based attack patterns, which traditional ML models often miss.

- **High performance on structured IoT data**  
  Achieves strong results on TON_IoT, showing suitability for IoT-driven environments.

- **Event-driven efficiency**  
  Computation occurs only when spikes are generated, leading to highly sparse (~99%) and efficient processing.

- **Energy-efficient design**  
  Low spike activity results in reduced energy consumption, making it suitable for edge devices.

- **Dataset sensitivity**  
  Performance varies across datasets, highlighting the importance of data quality and feature representation.

- **Competitive with traditional models**  
  Outperforms baseline models on TON_IoT, demonstrating the potential of neuromorphic approaches in cybersecurity.
  
------------------------------------------------------------

## 17. ⚠️ Limitations

- **Lower performance on CIC dataset**  
  The model performs better on TON_IoT compared to CIC-IDS2018 due to more complex and noisy traffic patterns in CIC.

- **Higher latency compared to traditional ML**  
  The use of Brian2 simulation and time-based spike processing increases inference time.

- **Unsupervised learning limitation**  
  STDP does not directly use labels, requiring post-training mapping for classification.

- **Dependence on spike encoding**  
  Model performance is sensitive to encoding strategy (TTFS) and threshold selection.

- **Limited model complexity**  
  The current SNN architecture is simple and may not capture highly complex attack behaviors.

- **No real-world deployment validation**  
  The system is tested in simulation and not yet validated on actual edge devices.
  

------------------------------------------------------------
---

## 👥 Contributors

Rakesh · Ujwal · Prajwal · Tarun  

Department of Computer Science and Engineering (Cyber Security)  
School of Engineering, Dayananda Sagar University  

---

## 🧑‍🏫 Mentor

Dr. Prajwalasimha S N,  
Associate Professor  

Department of Computer Science and Engineering (Cyber Security)  
School of Engineering, Dayananda Sagar University  

---

## 🔬 Laboratory

TTEH LAB · School of Engineering · Dayananda Sagar University  

_Bangalore – 562112, Karnataka, India_

---

------------------------------------------------------------

## Project Summary

This project demonstrates a complete neuromorphic IDS using:
- Spiking Neural Networks
- Event-driven computation
- Temporal learning

It highlights the potential of brain-inspired AI for efficient cybersecurity in IoT environments.
