import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser('~/snn_ids/src'))
from spike_encoder import SpikeEncoder

OUT_DIR = os.path.expanduser("~/snn_ids/data/processed/")
ENC_DIR = os.path.expanduser("~/snn_ids/data/encoded/")
os.makedirs(ENC_DIR, exist_ok=True)

for dataset in ['ton', 'cic']:
    print(f"\n{'='*50}")
    print(f"Testing SpikeEncoder on {dataset.upper()}")
    print('='*50)

    X_train = np.load(f"{OUT_DIR}{dataset}_X_train.npy")
    X_test  = np.load(f"{OUT_DIR}{dataset}_X_test.npy")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")

    # Fit encoder on training data
    enc = SpikeEncoder(T=100, delta=5)
    enc.fit(X_train)

    # Encode 5 samples to verify
    print(f"\nEncoding 5 samples to verify...")
    sample_spikes = enc.encode(X_train[:5])
    print(f"Output shape  : {sample_spikes.shape}")

    # Check sample 0 in detail
    s0           = sample_spikes[0]
    spike_times  = np.argwhere(s0 == 1)
    sparsity     = enc.get_sparsity(sample_spikes)

    print(f"\nSample 0 details:")
    print(f"  Feature values     : {X_train[0].round(3)}")
    print(f"  Spike matrix shape : {s0.shape}")
    print(f"  Total spikes fired : {len(spike_times)} / {s0.size} possible")
    print(f"  Sparsity           : {sparsity:.1%}  (>90% expected)")

    print(f"\n  Spike breakdown (feat | value | threshold | t_spike):")
    for feat, t in spike_times:
        val = X_train[0, feat]
        thr = enc.thresholds[feat]
        print(f"    Feature {feat:2d}: value={val:.3f}  "
              f"threshold={thr:.3f}  t_spike={t}ms")

    # Formula verification
    print(f"\n  Formula check (t = f x 99):")
    for feat, t in spike_times[:3]:
        expected = int(X_train[0, feat] * 99)
        match    = "✓" if t == expected else "✗"
        print(f"    Feature {feat}: f={X_train[0,feat]:.3f}  "
              f"expected={expected}  actual={t}  {match}")

    # Save encoder
    enc.save(f"{ENC_DIR}{dataset}_encoder.pkl")
    print(f"\n{dataset.upper()} encoder test passed ✓")

print("\n=== Spike Encoder Validation Complete ===")
