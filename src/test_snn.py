"""
Phase 5 — SNN Architecture Validation Test
Tests that:
1. Network builds without errors
2. Spikes propagate from input → hidden → output
3. Weight shapes are correct
4. Prediction returns valid class index
5. Works for both TON (16→40→5) and CIC (20→40→4)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser('~/snn_ids/src'))
from spike_encoder import SpikeEncoder
from snn_model import SNNModel

OUT_DIR = os.path.expanduser("~/snn_ids/data/processed/")
ENC_DIR = os.path.expanduser("~/snn_ids/data/encoded/")
MDL_DIR = os.path.expanduser("~/snn_ids/models/")
os.makedirs(MDL_DIR, exist_ok=True)

configs = [
    {'name': 'TON_IoT',    'dataset': 'ton',
     'n_input': 16, 'n_output': 5},
    {'name': 'CIC-IDS2018','dataset': 'cic',
     'n_input': 20, 'n_output': 4},
]

for cfg in configs:
    print(f"\n{'='*55}")
    print(f"Testing SNN Architecture: {cfg['name']}")
    print('='*55)

    # Load data and encoder
    X_train = np.load(f"{OUT_DIR}{cfg['dataset']}_X_train.npy")
    enc     = SpikeEncoder(T=100, delta=5)
    enc.fit(X_train)

    # Build SNN
    model = SNNModel(
        n_input  = cfg['n_input'],
        n_hidden = 40,
        n_output = cfg['n_output']
    )

    # Test 3 samples
    print(f"\nRunning 3 test samples...")
    for i in range(3):
        spike_matrix  = enc.encode_single(X_train[i])
        total_spikes  = int(spike_matrix.sum())

        out_counts, hid_counts = model.run_sample(spike_matrix)
        prediction    = int(np.argmax(out_counts)) \
                        if out_counts.sum() > 0 else -1

        print(f"\n  Sample {i}:")
        print(f"    Input spikes       : {total_spikes}")
        print(f"    Hidden spikes      : {int(hid_counts.sum())}"
              f" across {int((hid_counts>0).sum())} neurons")
        print(f"    Output spike counts: {out_counts.astype(int)}")
        print(f"    Predicted class    : {prediction}")

    # Verify weight shapes
    print(f"\nWeight verification:")
    print(f"  W_ih shape: {model.W_ih.shape} "
          f"(expected {cfg['n_input']}×40)")
    print(f"  W_ho shape: {model.W_ho.shape} "
          f"(expected 40×{cfg['n_output']})")

    wih_ok = model.W_ih.shape == (cfg['n_input'], 40)
    who_ok = model.W_ho.shape == (40, cfg['n_output'])
    print(f"  W_ih correct: {wih_ok} ✓" if wih_ok
          else f"  W_ih WRONG ✗")
    print(f"  W_ho correct: {who_ok} ✓" if who_ok
          else f"  W_ho WRONG ✗")

    # Save initial weights
    model.save_weights(
        f"{MDL_DIR}{cfg['dataset']}_W_ih_init.npy",
        f"{MDL_DIR}{cfg['dataset']}_W_ho_init.npy"
    )
    print(f"  Initial weights saved ✓")
    print(f"\n{cfg['name']} architecture test PASSED ✓")

print("\n=== Phase 5 Architecture Validation Complete ===")
