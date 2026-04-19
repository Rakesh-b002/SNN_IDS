# Documented Deviations from Paper

This document records all differences between our implementation
and the original paper specification, with full justification.

| # | Parameter | Paper | Ours | Reason |
|---|---|---|---|---|
| 1 | TON features | 20 | 16 | Kaggle subset has fewer numeric cols |
| 2 | Spike threshold θ | mean | 0.5×mean | Log-transformed data needs lower threshold |
| 3 | TON scaling | MinMax | log1p+MinMax | Extreme outliers in byte-count features |
| 4 | CIC scaling | MinMax | RobustScaler | Same outlier problem |
| 5 | Readout layer | WTA | RandomForest | Unsupervised WTA collapsed without lateral inhibition |
| 6 | Training size | 2.3M | 168k (TON) | RAM constraint on 8GB VM |
| 7 | CIC classes | 5 | 4 | PortScan absent in sampled subset |
| 8 | Latency | 6.2ms | 132ms | Software simulation vs neuromorphic hardware |
| 9 | Energy | 3.9mJ | 1.38nJ | Different measurement method |

## Impact on Results
- TON_IoT: 96.2% accuracy (paper: 96.5%) — within 0.3%
- CIC-IDS2018: 77.3% accuracy — gap due to class imbalance
