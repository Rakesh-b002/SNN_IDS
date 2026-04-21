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

Modern IoT deployments face a fundamental tension: security demands continuous, low-latency traffic inspection, while edge hardware imposes strict constraints on energy, memory, and compute. Conventional deep learning IDS models are accurate but energetically expensive, often requiring GPU-class hardware.
Spiking Neural Networks address this by processing information through sparse, event-driven spike trains — closely mimicking biological neural computation. Key properties that make SNNs attractive for this problem:

Event-driven computation → near-zero cost when no spikes are fired
Ultra-low inference energy → sub-nanojoule-range on neuromorphic hardware
Temporal encoding → naturally suited to time-series network traffic
Bio-plausible learning → STDP enables unsupervised feature extraction without backpropagation

The core research question is: Can a biologically-inspired SNN achieve competitive intrusion detection accuracy with energy consumption several orders of magnitude below conventional classifiers?

------------------------------------------------------------

## 3. Proposed System Architecture


| Stage | Component | Description | Output |
|------|----------|------------|--------|
| 1 | Data Processing | Cleans data, handles missing values, performs feature selection and normalization | Structured feature vectors |
| 2 | Spike Encoding (TTFS) | Converts features into spike timings based on magnitude | Sparse spike trains (~99% sparsity) |
| 3 | SNN Model | Processes spike data using LIF neurons (Input → Hidden → Output) | Temporal spike activity |
| 4 | STDP Learning | Updates synaptic weights based on spike timing relationships | Learned model parameters |
| 5 | Classification | Determines class based on output neuron spike activity | Predicted attack class |
| 6 | Evaluation | Computes accuracy, F1 score, latency, and energy; generates visual outputs | Metrics and result artifacts (`results/`) |

------------------------------------------------------------

## 4. 🔄 End-to-End Workflow

System Flow:

Raw Data → Preprocessing → Feature Selection → Normalization → Spike Encoding → SNN Processing → STDP Training → Classification → Evaluation

------------------------------------------------------------
```bash
Step   Stage                Description                                              Output
----------------------------------------------------------------------------------------------
1      Data Collection      Load TON_IoT and CIC-IDS2018 datasets                   Raw network data
2      Preprocessing        Clean data, handle missing/inf values, map labels       Cleaned dataset
3      Feature Selection    Select relevant features (MI + RFE)                     Reduced features
4      Normalization        Scale features to uniform range                         Normalized data
5      Spike Encoding       Convert features into spike timings (TTFS)              Sparse spike trains
6      SNN Processing       Process spikes using LIF neurons                        Temporal spike activity
7      STDP Training        Update weights based on spike timing                    Trained SNN model
8      Classification       Predict class using output neuron spikes                Predicted labels
9      Evaluation           Compute metrics and generate results                    Metrics + results/
```
------------------------------------------------------------

## 5. 📊 Datasets
```
Dataset         Samples     Features        Description                         Link
----------------------------------------------------------------------------------------------
TON_IoT         ~2.3M       44 → 16         IoT telemetry attack dataset        https://www.kaggle.com/datasets/dhoogla/csecicids2018
CIC-IDS2018     ~3.1M       78 → 20         Enterprise network attack dataset   https://www.kaggle.com/datasets/dhoogla/nftoniotv2
```
----------------------------------------------------------------------------------------------

## 📂 Dataset Structure

After downloading, organize the datasets as follows:
```
data/
├── ton_iot/
│   └── (TON_IoT CSV files)
├── cic_ids/
│   └── (CIC-IDS2018 CSV files)
```
----------------------------------------------------------------------------------------------

## 🔄 Dataset Processing

Each dataset undergoes the following steps:

- Removal of missing and duplicate values
- Handling of Inf / -Inf values
- Label mapping into 5 classes:
  Normal | DoS | DDoS | Scan | Others
- Feature selection (MI + RFE)
- Normalization (MinMax scaling)

----------------------------------------------------------------------------------------------

## ⚠️ Notes

- Ensure sufficient storage (~5–8 GB total for both datasets)
- CIC-IDS2018 may require chunk processing due to large size
- Verify dataset paths in:
  src/preprocess.py
  src/preprocess_cic.py

------------------------------------------------------------

## 🔹 SECTION 6 — Data Preprocessing

Before spike-based processing, raw network traffic data is cleaned and structured.

Both datasets — TON_IoT and CIC-IDS2018 — undergo the following steps:

- Data Cleaning
  - Remove missing values and duplicates
  - Handle infinite values and inconsistencies

- Label Mapping
  - Map all attack types into 5 unified classes:
    Normal / DoS / DDoS / Scan / Others

- Feature Selection
  - Use Mutual Information (MI) and Recursive Feature Elimination (RFE)
  - Reduce features:
    TON_IoT: 44 → 16
    CIC-IDS2018: 78 → 20

- Normalization
  - TON_IoT → log1p + MinMaxScaler
  - CIC-IDS2018 → RobustScaler (handles outliers)

Output: Clean, normalized feature matrix ready for spike encoding.


## ⚡ SECTION 7 — Spike Encoding (TTFS)

SNNs operate on spikes instead of continuous values.

We use Time-To-First-Spike (TTFS):

t_spike = feature_value × 99 ms

- Higher values → earlier spikes
- Lower values → later spikes

Encoding rules:
- Spike only if:
  feature_value ≥ 0.5 × mean(feature)
- Else → no spike

Properties:
- Single spike per feature
- Time window = 100 ms
- Refractory period = 5 ms

Result:
- TON_IoT sparsity = 99.9%
- CIC sparsity = 99.7%

This sparsity directly enables energy efficiency.


## 🧠 SECTION 8 — SNN Model Architecture

Three-layer architecture:

Input Layer
- 16–20 neurons (one per feature)
- Passes spike timings

Hidden Layer (40 LIF neurons)

Neuron model:
τ · dV/dt = -V + Σ(w × spikes)

Parameters:
- τ = 10 ms
- Threshold = 1.0
- Reset = 0
- Refractory = 5 ms

Key mechanisms:
- Competitive Inhibition → first neuron suppresses others
- Homeostatic Adaptation → balances neuron activity

Output Layer
- 5 neurons:
  Normal / DoS / DDoS / Scan / Others
- Final prediction = highest spike activity

Hybrid Extension:
- Extract 96-dimensional spike feature vector
- Pass to Random Forest for classification


## 🔁 SECTION 9 — Learning Mechanism (STDP)

STDP is used for unsupervised learning.

Weight update:

Δw = A+ · exp(-Δt / τ+)    if Δt > 0
Δw = -A- · exp(Δt / τ-)    if Δt < 0

Where:
- Δt = t_post − t_pre
- A± = 0.01
- τ± = 20 ms

Interpretation:
- Pre before post → strengthen connection
- Pre after post → weaken connection

Key advantages:
- No backpropagation
- Local learning rule
- Captures temporal causality

Different attacks produce distinct spike timing patterns, which are learned automatically.


## 🚀 SECTION 10 — Training Process

Training is performed over 30 epochs.

For each sample:
1. Encode features into spike trains
2. Run 100 ms SNN simulation (Brian2)
3. Update weights using STDP
4. Apply homeostatic adjustments
5. Move to next sample

After training:
- Extract spike-based features from hidden layer
- Train Random Forest using labels (supervised step)

Final model artifacts:
- W_ih_trained.npy   → Input-to-hidden weights
- W_ho_trained.npy   → Hidden-to-output weights
- readout_rich.pkl   → Random Forest classifier

The system is now ready for inference.

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
```bash
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
```
------------------------------------------------------------

## 14. 🗂️ Code Architecture
```bash
snn_ids/
├── docs/                           # 📄 Documentation and notes
│   └── deviations.md               # Project deviations from paper
│
├── results/                        # 📊 Outputs and evaluation results
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   ├── latency_comparison.png
│   ├── energy_proxy.png
│   ├── confusion_matrices.png
│   ├── confusion_ton.png
│   ├── confusion_cic.png
│   ├── per_class_f1.png
│   ├── ton_results.json
│   └── cic_results.json
│
├── sample_data/                    # 🧪 Small sample datasets for testing
│   ├── ton_sample.npy
│   ├── ton_labels.npy
│   ├── cic_sample.npy
│   └── cic_labels.npy
│
├── src/                            # 🧠 Core implementation
│   ├── preprocess.py
│   ├── preprocess_cic.py
│   ├── spike_encoder.py
│   ├── snn_model.py
│   ├── train_stdp.py
│   ├── evaluate.py
│   ├── test_encoder.py
│   └── test_snn.py
│
├── README.md
└── .gitignore
```
------------------------------------------------------------

## 15. 🚀 Setup & Usage

### ⚙️ Requirements
```bash
Component        Requirement
----------------------------------------
Python           3.9+ (3.10 recommended)
RAM              4 GB minimum
Libraries        brian2, numpy, pandas, scikit-learn, matplotlib
Datasets         TON_IoT, CIC-IDS2018
```
----------------------------------------

## 📦 Installation

# 1. Clone repository
```bash
git clone <REPO_URL>
cd <REPO_ROOT>
```
# 2. Create virtual environment
python -m venv snn_env

# 3. Activate environment
# Windows
snn_env\Scripts\activate
# Linux / macOS
source snn_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

----------------------------------------

## 📂 Dataset Setup

Download datasets:
- TON_IoT
- CIC-IDS2018

Place them in:
```bash
data/
├── ton_iot/
├── cic_ids/
```
Update dataset paths in:
- src/preprocess.py
- src/preprocess_cic.py

----------------------------------------

## ▶️ Quick Start

# Step 1: Preprocess data
python src/preprocess.py
python src/preprocess_cic.py

# Step 2: Validate spike encoding
python src/test_encoder.py

# Step 3: Train SNN (STDP)
python src/train_stdp.py

# Step 4: Evaluate model
python src/evaluate.py


----------------------------------------

## 🧪 Output

After running the pipeline, you will get:

- Processed datasets
- Spike-encoded data
- Trained model weights (.npy)
- Evaluation metrics:
  Accuracy, F1 Score, Latency, Energy

----------------------------------------

## 🛠️ Troubleshooting

No spikes generated:
→ Check normalization and threshold values

Low accuracy:
→ Verify feature selection and encoding

Slow execution:
→ Reduce simulation window or dataset size

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

Rakesh · Ujwal · Prajwal · Tarun · Dhruva

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
