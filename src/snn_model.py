"""
SNN Architecture — Phase 5
3-layer LIF network built with Brian2

Architecture: N_input → 40 hidden LIF → N_output LIF
Paper: Event-Driven IDS using SNN (ICSCSA-2025)
LIF equation: tau_m × dV/dt = -V + R×I  (Paper Eq. 4)
Parameters: tau_m=10ms, Vth=1.0, Vreset=0, refractory=5ms
"""

import numpy as np
import os
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor,
    SpikeGeneratorGroup, Network,
    ms, defaultclock, start_scope, prefs
)

prefs.codegen.target = 'numpy'

class SNNModel:
    def __init__(self, n_input, n_hidden=40, n_output=5,
                 tau_m=10, Vth=1.0, Vreset=0.0, refractory=5):
        self.n_input    = n_input
        self.n_hidden   = n_hidden
        self.n_output   = n_output
        self.tau_m      = tau_m
        self.Vth        = Vth
        self.Vreset     = Vreset
        self.refractory = refractory

        # W_ih: input→hidden  (trained by STDP in Phase 6)
        # W_ho: hidden→output (fixed — assigned after STDP)
        self.W_ih = np.random.uniform(0.0, 0.5,
                    (n_input, n_hidden)).astype(np.float32)
        self.W_ho = np.random.uniform(0.0, 0.5,
                    (n_hidden, n_output)).astype(np.float32)

        print(f"SNNModel initialized:")
        print(f"  Architecture : {n_input} → {n_hidden} → {n_output}")
        print(f"  tau_m        : {tau_m}ms")
        print(f"  Threshold    : {Vth}")
        print(f"  Reset        : {Vreset}")
        print(f"  Refractory   : {refractory}ms")
        print(f"  W_ih shape   : {self.W_ih.shape}")
        print(f"  W_ho shape   : {self.W_ho.shape}")

    def run_sample(self, spike_matrix, duration=100):
        """
        Run SNN simulation for one sample.

        Args:
            spike_matrix: 2D array (n_input, T) from SpikeEncoder
            duration    : simulation time in ms (100ms paper spec)

        Returns:
            output_counts: 1D array (n_output,) spike counts per class
            hidden_counts: 1D array (n_hidden,) spike counts per neuron
        """
        start_scope()

        # ── Convert spike matrix to Brian2 event lists ────────────
        rows, cols = np.where(spike_matrix == 1.0)
        input_indices = rows.astype(np.int32)
        input_times   = cols.astype(np.float64)

        if len(input_indices) == 0:
            return np.zeros(self.n_output), np.zeros(self.n_hidden)

        # ── Layer 1: Input ─────────────────────────────────────────
        input_layer = SpikeGeneratorGroup(
            N       = self.n_input,
            indices = input_indices,
            times   = input_times * ms
        )

        # ── LIF equation ───────────────────────────────────────────
        # w defined as synapse parameter — NOT in neuron model
        lif_eqs = '''
            dV/dt = -V / (tau_m * ms) : 1 (unless refractory)
            tau_m : 1 (shared)
        '''

        # ── Layer 2: Hidden (40 LIF neurons) ──────────────────────
        hidden_layer = NeuronGroup(
            N          = self.n_hidden,
            model      = lif_eqs,
            threshold  = f'V > {self.Vth}',
            reset      = f'V = {self.Vreset}',
            refractory = f'{self.refractory}*ms',
            method     = 'euler'
        )
        hidden_layer.V     = 0.0
        hidden_layer.tau_m = self.tau_m

        # ── Layer 3: Output ────────────────────────────────────────
        output_layer = NeuronGroup(
            N          = self.n_output,
            model      = lif_eqs,
            threshold  = f'V > {self.Vth}',
            reset      = f'V = {self.Vreset}',
            refractory = f'{self.refractory}*ms',
            method     = 'euler'
        )
        output_layer.V     = 0.0
        output_layer.tau_m = self.tau_m

        # ── Synapses Input → Hidden ────────────────────────────────
        # w is defined as a synapse state variable here
        syn_ih = Synapses(
            input_layer, hidden_layer,
            model    = 'w : 1',        # w is the weight variable
            on_pre   = 'V_post += w'   # spike adds w to V directly
        )
        syn_ih.connect()  # all-to-all: each input → each hidden

        # Set weights from W_ih matrix
        # syn_ih.w[i,j] = weight from input i to hidden j
        syn_ih.w = self.W_ih.flatten()

        # ── Synapses Hidden → Output ───────────────────────────────
        syn_ho = Synapses(
            hidden_layer, output_layer,
            model  = 'w : 1',
            on_pre = 'V_post += w'
        )
        syn_ho.connect()
        syn_ho.w = self.W_ho.flatten()

        # ── Spike Monitors ─────────────────────────────────────────
        hidden_monitor = SpikeMonitor(hidden_layer)
        output_monitor = SpikeMonitor(output_layer)

        # ── Run ────────────────────────────────────────────────────
        net = Network(
            input_layer, hidden_layer, output_layer,
            syn_ih, syn_ho,
            hidden_monitor, output_monitor
        )
        net.run(duration * ms)

        # ── Count spikes ───────────────────────────────────────────
        hidden_counts = np.zeros(self.n_hidden)
        for nid in hidden_monitor.i:
            hidden_counts[int(nid)] += 1

        output_counts = np.zeros(self.n_output)
        for nid in output_monitor.i:
            output_counts[int(nid)] += 1

        return output_counts, hidden_counts

    def predict(self, spike_matrix):
        """Predict class — argmax of output spike counts."""
        output_counts, _ = self.run_sample(spike_matrix)
        if output_counts.sum() == 0:
            return -1   # no output spikes
        return int(np.argmax(output_counts))

    def save_weights(self, path_ih, path_ho):
        np.save(path_ih, self.W_ih)
        np.save(path_ho, self.W_ho)
        print(f"Weights saved: {path_ih}, {path_ho}")

    def load_weights(self, path_ih, path_ho):
        self.W_ih = np.load(path_ih)
        self.W_ho = np.load(path_ho)
        print(f"Weights loaded: {path_ih}, {path_ho}")
