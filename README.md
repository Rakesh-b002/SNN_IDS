# SNN-Based Intrusion Detection System

Implementation of Event-Driven IDS using Spiking Neural Networks
for Edge and IoT Security.


## Architecture
- Input: 16 (TON_IoT) / 20 (CIC-IDS2018) neurons
- Hidden: 40 LIF neurons
- Output: 5 (TON) / 4 (CIC) neurons
- Learning: STDP (unsupervised)
- Simulator: Brian2 v2.7.1

## Datasets
- TON_IoT Network Dataset (Kaggle)
- CIC-IDS2018 (Canadian Institute for Cybersecurity)

## Progress
- [x] Phase 1: Environment Setup
- [x] Phase 2: Dataset Acquisition
- [x] Phase 3: Data Preprocessing
- [x] Phase 4: Spike Encoding
- [ ] Phase 5: SNN Architecture
- [ ] Phase 6: STDP Training
- [ ] Phase 7: Evaluation
