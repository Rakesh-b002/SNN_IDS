"""
Phase 6 — STDP Training with Competitive Learning (Fixed)
Uses winner-take-all STDP: only winning hidden neuron updated per sample.
This prevents the collapse problem where all neurons specialize to one class.
"""

import numpy as np
import joblib
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/snn_ids/src'))
from spike_encoder import SpikeEncoder
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor,
    SpikeGeneratorGroup, Network,
    ms, start_scope, prefs
)
prefs.codegen.target = 'numpy'

BASE    = os.path.expanduser('~/snn_ids')
OUT_DIR = f"{BASE}/data/processed"
MDL_DIR = f"{BASE}/models"
os.makedirs(MDL_DIR, exist_ok=True)

# ── STDP Parameters (Paper Table II) ──────────────────────────
A_PLUS    = 0.01
A_MINUS   = 0.01
TAU_PLUS  = 20.0
TAU_MINUS = 20.0
W_MIN     = 0.0
W_MAX     = 1.0
N_EPOCHS  = 30

# ── Competitive STDP Update ────────────────────────────────────
def stdp_update_competitive(W, pre_times, post_times, n_hidden):
    """
    Winner-take-all STDP: only update the FIRST firing hidden neuron.
    This prevents collapse — each neuron competes for different inputs.
    """
    if not post_times:
        return W, 0

    # Find the winner = hidden neuron that fired earliest
    winner = min(post_times, key=post_times.get)
    t_post = post_times[winner]
    n_updates = 0

    for i, t_pre in pre_times.items():
        dt = t_post - t_pre
        if dt > 0:
            dW = A_PLUS * np.exp(-dt / TAU_PLUS)
            W[i, winner] = min(W[i, winner] + dW, W_MAX)
            n_updates += 1
        elif dt < 0:
            dW = A_MINUS * np.exp(dt / TAU_MINUS)
            W[i, winner] = max(W[i, winner] - dW, W_MIN)
            n_updates += 1

    return W, n_updates

# ── Weight Normalization ───────────────────────────────────────
def normalize_weights(W):
    """
    L1 normalize each hidden neuron's incoming weights.
    Keeps total input strength equal across all neurons.
    Prevents any single neuron from dominating.
    """
    n_pre, n_post = W.shape
    target = n_pre * 0.25
    for j in range(n_post):
        s = W[:, j].sum()
        if s > target * 0.8:
            W[:, j] *= target / s
    return W

# ── Homeostatic threshold adaptation ──────────────────────────
def adapt_thresholds(fire_rates, base_threshold=1.0,
                     target_rate=0.1, adapt_rate=0.01):
    """
    Neurons that fire too much get higher threshold (harder to fire).
    Neurons that fire too little get lower threshold (easier to fire).
    This ensures all neurons get a chance to specialize.
    target_rate = fraction of samples each neuron should fire for
    """
    thresholds = np.ones(len(fire_rates)) * base_threshold
    for j in range(len(fire_rates)):
        if fire_rates[j] > target_rate:
            thresholds[j] = base_threshold + adapt_rate * \
                            (fire_rates[j] - target_rate)
        elif fire_rates[j] < target_rate * 0.5:
            thresholds[j] = max(0.5, base_threshold - adapt_rate)
    return thresholds

# ── Run one sample ─────────────────────────────────────────────
def run_sample_timing(spike_matrix, W_ih, W_ho,
                       n_input, n_hidden, n_output,
                       thresholds=None,
                       tau_m=10, Vreset=0.0,
                       refractory=5, duration=100):
    start_scope()
    rows, cols = np.where(spike_matrix == 1.0)
    if len(rows) == 0:
        return {}, {}, np.zeros(n_output), np.zeros(n_hidden)

    input_indices = rows.astype(np.int32)
    input_times   = cols.astype(np.float64)

    pre_times = {}
    for idx, t in zip(input_indices, input_times):
        if int(idx) not in pre_times:
            pre_times[int(idx)] = float(t)

    input_layer = SpikeGeneratorGroup(
        N=n_input, indices=input_indices,
        times=input_times * ms)

    # Use per-neuron thresholds if provided
    if thresholds is None:
        thresholds = np.ones(n_hidden)

    lif_eqs = '''
        dV/dt = -V / (tau_m * ms) : 1 (unless refractory)
        tau_m : 1 (shared)
        Vth   : 1
    '''

    hidden_layer = NeuronGroup(
        N=n_hidden, model=lif_eqs,
        threshold='V > Vth', reset=f'V = {Vreset}',
        refractory=f'{refractory}*ms', method='euler')
    hidden_layer.V     = 0.0
    hidden_layer.tau_m = tau_m
    hidden_layer.Vth   = thresholds

    lif_eqs_out = '''
        dV/dt = -V / (tau_m * ms) : 1 (unless refractory)
        tau_m : 1 (shared)
    '''
    output_layer = NeuronGroup(
        N=n_output, model=lif_eqs_out,
        threshold='V > 1.0', reset=f'V = {Vreset}',
        refractory=f'{refractory}*ms', method='euler')
    output_layer.V     = 0.0
    output_layer.tau_m = tau_m

    syn_ih = Synapses(input_layer, hidden_layer,
                      model='w : 1', on_pre='V_post += w')
    syn_ih.connect()
    syn_ih.w = W_ih.flatten()

    syn_ho = Synapses(hidden_layer, output_layer,
                      model='w : 1', on_pre='V_post += w')
    syn_ho.connect()
    syn_ho.w = W_ho.flatten()

    hidden_mon = SpikeMonitor(hidden_layer)
    output_mon = SpikeMonitor(output_layer)

    net = Network(input_layer, hidden_layer, output_layer,
                  syn_ih, syn_ho, hidden_mon, output_mon)
    net.run(duration * ms)

    post_times = {}
    for nid, t in zip(hidden_mon.i, hidden_mon.t / ms):
        if int(nid) not in post_times:
            post_times[int(nid)] = float(t)

    hid_counts = np.zeros(n_hidden)
    for nid in hidden_mon.i:
        hid_counts[int(nid)] += 1

    out_counts = np.zeros(n_output)
    for nid in output_mon.i:
        out_counts[int(nid)] += 1

    return pre_times, post_times, out_counts, hid_counts

# ── Winner-take-all class assignment ──────────────────────────
def assign_classes(W_ih, X_train, y_train, encoder,
                    n_input, n_hidden, n_output,
                    thresholds, n_samples=2000):
    print(f"  Winner-take-all assignment ({n_samples} samples)...")
    W_ho_temp   = np.ones((n_hidden, n_output),
                           dtype=np.float32) * 0.25
    fire_counts = np.zeros((n_hidden, n_output))

    per_class = n_samples // n_output
    indices   = []
    for c in range(n_output):
        cls_idx = np.where(y_train == c)[0]
        chosen  = np.random.choice(cls_idx,
                                    min(per_class, len(cls_idx)),
                                    replace=False)
        indices.extend(chosen)
    np.random.shuffle(indices)

    for i in tqdm(indices, desc="  Assigning",
                  leave=False, ncols=70):
        sm = encoder.encode_single(X_train[i])
        if sm.sum() == 0:
            continue
        _, _, _, hid = run_sample_timing(
            sm, W_ih, W_ho_temp,
            n_input, n_hidden, n_output, thresholds)
        fire_counts[:, int(y_train[i])] += hid

    assignments = np.argmax(fire_counts, axis=1)
    W_ho = np.zeros((n_hidden, n_output), dtype=np.float32)
    for h in range(n_hidden):
        W_ho[h, assignments[h]] = 1.0

    counts = np.bincount(assignments, minlength=n_output)
    print(f"  Neurons per class: {counts}")

    # Ensure every class has at least 1 neuron
    unassigned = np.where(counts == 0)[0]
    if len(unassigned) > 0:
        print(f"  Fixing unassigned classes: {unassigned}")
        over = np.argsort(counts)[::-1]
        for uc in unassigned:
            donor_neurons = np.where(assignments == over[0])[0]
            give = donor_neurons[:max(2, len(donor_neurons)//4)]
            for h in give:
                assignments[h] = uc
                W_ho[h, over[0]] = 0.0
                W_ho[h, uc]      = 1.0
        counts = np.bincount(assignments, minlength=n_output)
        print(f"  After fix: {counts}")

    return W_ho, assignments

# ══════════════════════════════════════════════════════════════
def train_dataset(dataset_name, n_input, n_output,
                  samples_per_epoch=5000):
    print(f"\n{'='*60}")
    print(f"STDP TRAINING (Competitive): {dataset_name.upper()}")
    print(f"Architecture : {n_input} → 40 → {n_output}")
    print(f"Epochs       : {N_EPOCHS}")
    print(f"Samples/epoch: {samples_per_epoch}")
    print('='*60)

    X_train = np.load(f"{OUT_DIR}/{dataset_name}_X_train.npy")
    y_train = np.load(f"{OUT_DIR}/{dataset_name}_y_train.npy")
    X_test  = np.load(f"{OUT_DIR}/{dataset_name}_X_test.npy")
    y_test  = np.load(f"{OUT_DIR}/{dataset_name}_y_test.npy")
    le      = joblib.load(f"{OUT_DIR}/{dataset_name}_label_encoder.pkl")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes: {le.classes_}")

    encoder = SpikeEncoder(T=100, delta=5)
    encoder.fit(X_train)

    n_hidden = 40
    W_ih = np.random.uniform(0.0, 0.5,
            (n_input, n_hidden)).astype(np.float32)
    W_ho = np.random.uniform(0.0, 0.5,
            (n_hidden, n_output)).astype(np.float32)

    # Adaptive per-neuron thresholds
    thresholds  = np.ones(n_hidden, dtype=np.float32)
    fire_rates  = np.zeros(n_hidden, dtype=np.float32)

    n_classes   = n_output
    total_start = time.time()
    print(f"\nStarting competitive STDP training...")

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start   = time.time()
        total_updates = 0
        silent_count  = 0
        epoch_fires   = np.zeros(n_hidden)

        # Stratified sample
        indices = []
        per_class = samples_per_epoch // n_classes
        for c in range(n_classes):
            cls_idx = np.where(y_train == c)[0]
            chosen  = np.random.choice(cls_idx,
                                        min(per_class, len(cls_idx)),
                                        replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        np.random.shuffle(indices)

        for i in tqdm(indices,
                      desc=f"  Epoch {epoch:2d}/{N_EPOCHS}",
                      leave=False, ncols=70):
            sm = encoder.encode_single(X_train[i])
            if sm.sum() == 0:
                silent_count += 1
                continue

            pre_t, post_t, _, hid_counts = run_sample_timing(
                sm, W_ih, W_ho,
                n_input, n_hidden, n_output, thresholds)

            epoch_fires += hid_counts

            # Competitive STDP — winner only
            if len(post_t) > 0:
                W_ih, n_upd = stdp_update_competitive(
                    W_ih, pre_t, post_t, n_hidden)
                total_updates += n_upd

        # Normalize weights
        W_ih = normalize_weights(W_ih)

        # Update adaptive thresholds
        fire_rates  = epoch_fires / len(indices)
        thresholds  = adapt_thresholds(fire_rates).astype(np.float32)

        epoch_time  = time.time() - epoch_start
        active_neurons = (epoch_fires > 0).sum()
        w_mean      = W_ih.mean()
        w_max       = W_ih.max()

        print(f"  Epoch {epoch:2d}/{N_EPOCHS} | "
              f"Updates: {total_updates:5d} | "
              f"W mean={w_mean:.4f} max={w_max:.4f} | "
              f"Active neurons: {active_neurons}/40 | "
              f"Time: {epoch_time:.0f}s")

        if epoch % 5 == 0:
            np.save(f"{MDL_DIR}/{dataset_name}_W_ih_v2_epoch{epoch}.npy",
                    W_ih.copy())
            print(f"  → Checkpoint: epoch {epoch}")

    total_time = time.time() - total_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes")

    # Class assignment
    print(f"\nRunning winner-take-all class assignment...")
    W_ho_final, assignments = assign_classes(
        W_ih, X_train, y_train, encoder,
        n_input, n_hidden, n_output,
        thresholds, n_samples=2000)

    # Save
    np.save(f"{MDL_DIR}/{dataset_name}_W_ih_trained.npy", W_ih)
    np.save(f"{MDL_DIR}/{dataset_name}_W_ho_trained.npy", W_ho_final)
    np.save(f"{MDL_DIR}/{dataset_name}_assignments.npy",  assignments)
    print(f"Weights saved.")

    # Quick validation
    print(f"\nQuick validation (200 test samples)...")
    correct = 0
    total   = 0
    val_idx = np.random.choice(len(X_test), 200, replace=False)

    for i in tqdm(val_idx, desc="  Validating",
                  leave=False, ncols=70):
        sm = encoder.encode_single(X_test[i])
        if sm.sum() == 0:
            continue
        _, _, out_counts, _ = run_sample_timing(
            sm, W_ih, W_ho_final,
            n_input, n_hidden, n_output, thresholds)
        if out_counts.sum() > 0:
            pred = int(np.argmax(out_counts))
            if pred == int(y_test[i]):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"  Quick accuracy: {acc:.1%} ({correct}/{total} with output)")
    return W_ih, W_ho_final, acc

# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both',
                        choices=['ton', 'cic', 'both'])
    parser.add_argument('--epochs',  type=int, default=N_EPOCHS)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    N_EPOCHS = args.epochs

    datasets = []
    if args.dataset in ['ton', 'both']:
        datasets.append(('ton', 16, 5, args.samples))
    if args.dataset in ['cic', 'both']:
        datasets.append(('cic', 20, 4, max(args.samples-1000, 4000)))

    results = {}
    for dname, n_in, n_out, n_samp in datasets:
        W_ih, W_ho, acc = train_dataset(dname, n_in, n_out, n_samp)
        results[dname] = acc

    print(f"\n{'='*60}")
    print("PHASE 6 COMPLETE")
    print('='*60)
    for dname, acc in results.items():
        print(f"  {dname.upper():12s}: quick val = {acc:.1%}")
    print("\nNext: python3 src/evaluate.py")


def train_resume(dataset_name, n_input, n_output,
                 resume_epoch, total_epochs=30,
                 samples_per_epoch=5000):
    """Resume training from a saved checkpoint."""
    print(f"\n{'='*60}")
    print(f"RESUMING: {dataset_name.upper()} from epoch {resume_epoch}")
    print(f"Remaining epochs: {resume_epoch+1} to {total_epochs}")
    print('='*60)

    X_train = np.load(f"{OUT_DIR}/{dataset_name}_X_train.npy")
    y_train = np.load(f"{OUT_DIR}/{dataset_name}_y_train.npy")
    X_test  = np.load(f"{OUT_DIR}/{dataset_name}_X_test.npy")
    y_test  = np.load(f"{OUT_DIR}/{dataset_name}_y_test.npy")
    le      = joblib.load(f"{OUT_DIR}/{dataset_name}_label_encoder.pkl")

    encoder = SpikeEncoder(T=100, delta=5)
    encoder.fit(X_train)

    n_hidden = 40

    # Load checkpoint weights
    ckpt = f"{MDL_DIR}/{dataset_name}_W_ih_v2_epoch{resume_epoch}.npy"
    W_ih = np.load(ckpt).astype(np.float32)
    W_ho = np.random.uniform(0.0, 0.5,
            (n_hidden, n_output)).astype(np.float32)
    print(f"Loaded: {ckpt}")
    print(f"W_ih shape={W_ih.shape} mean={W_ih.mean():.4f}")

    thresholds = np.ones(n_hidden, dtype=np.float32)
    fire_rates = np.zeros(n_hidden, dtype=np.float32)
    n_classes  = n_output
    total_start = time.time()

    for epoch in range(resume_epoch + 1, total_epochs + 1):
        epoch_start   = time.time()
        total_updates = 0
        silent_count  = 0
        epoch_fires   = np.zeros(n_hidden)

        indices = []
        per_class = samples_per_epoch // n_classes
        for c in range(n_classes):
            cls_idx = np.where(y_train == c)[0]
            chosen  = np.random.choice(cls_idx,
                                        min(per_class, len(cls_idx)),
                                        replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        np.random.shuffle(indices)

        for i in tqdm(indices,
                      desc=f"  Epoch {epoch:2d}/{total_epochs}",
                      leave=False, ncols=70):
            sm = encoder.encode_single(X_train[i])
            if sm.sum() == 0:
                silent_count += 1
                continue

            pre_t, post_t, _, hid_counts = run_sample_timing(
                sm, W_ih, W_ho,
                n_input, n_hidden, n_output, thresholds)

            epoch_fires += hid_counts

            if len(post_t) > 0:
                W_ih, n_upd = stdp_update_competitive(
                    W_ih, pre_t, post_t, n_hidden)
                total_updates += n_upd

        W_ih       = normalize_weights(W_ih)
        fire_rates = epoch_fires / len(indices)
        thresholds = adapt_thresholds(fire_rates).astype(np.float32)

        epoch_time     = time.time() - epoch_start
        active_neurons = (epoch_fires > 0).sum()
        print(f"  Epoch {epoch:2d}/{total_epochs} | "
              f"Updates: {total_updates:5d} | "
              f"W mean={W_ih.mean():.4f} max={W_ih.max():.4f} | "
              f"Active: {active_neurons}/40 | "
              f"Time: {epoch_time:.0f}s")

        if epoch % 5 == 0:
            np.save(f"{MDL_DIR}/{dataset_name}_W_ih_v2_epoch{epoch}.npy",
                    W_ih.copy())
            print(f"  → Checkpoint: epoch {epoch}")

    total_time = time.time() - total_start
    print(f"\nResume training complete in {total_time/60:.1f} minutes")

    # Class assignment
    print(f"\nRunning winner-take-all class assignment...")
    W_ho_final, assignments = assign_classes(
        W_ih, X_train, y_train, encoder,
        n_input, n_hidden, n_output,
        thresholds, n_samples=2000)

    np.save(f"{MDL_DIR}/{dataset_name}_W_ih_trained.npy", W_ih)
    np.save(f"{MDL_DIR}/{dataset_name}_W_ho_trained.npy", W_ho_final)
    np.save(f"{MDL_DIR}/{dataset_name}_assignments.npy",  assignments)
    print(f"Final weights saved.")

    # Quick validation
    print(f"\nQuick validation (200 test samples)...")
    correct = 0
    total   = 0
    val_idx = np.random.choice(len(X_test), 200, replace=False)

    for i in tqdm(val_idx, desc="  Validating",
                  leave=False, ncols=70):
        sm = encoder.encode_single(X_test[i])
        if sm.sum() == 0:
            continue
        _, _, out_counts, _ = run_sample_timing(
            sm, W_ih, W_ho_final,
            n_input, n_hidden, n_output, thresholds)
        if out_counts.sum() > 0:
            pred = int(np.argmax(out_counts))
            if pred == int(y_test[i]):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"  Quick accuracy: {acc:.1%} ({correct}/{total} with output)")
    return W_ih, W_ho_final, acc
