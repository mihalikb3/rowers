#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt
import shutil
import datetime

class BeadDataAnalyzer:
    def __init__(self):
        """Initialize the analyzer with default parameters"""
        self.time_series = None
        self.positions = None
        self.bead_positions = {}
        self.num_beads = 0
        self.dt = None
        
    def parse_filename(self, filename: str) -> dict:
        params = {}
        distances_match = re.search(r'\[(.*?)\]D', filename)
        if distances_match:
            params['distances'] = [float(x) for x in distances_match.group(1).split(',')]
        params['alpha'] = float(re.search(r'(\d+)alpha', filename).group(1))
        params['amplitude'] = float(re.search(r'(\d+\.\d+)A', filename).group(1))
        return params
        
    def read_data_file(self, filepath: str) -> None:
        data = np.loadtxt(filepath)
        self.time_series = data[:, 1]
        self.dt = self.time_series[1] - self.time_series[0]
        num_cols = data.shape[1]
        self.num_beads = (num_cols - 2) // 2
        self.positions = np.zeros((len(self.time_series), self.num_beads, 2))
        for i in range(self.num_beads):
            x_col = 2 + 2*i
            y_col = 3 + 2*i
            self.positions[:, i, 0] = data[:, x_col]
            self.positions[:, i, 1] = data[:, y_col]

class MultiBeadCoherenceAnalyzer:
    def __init__(self, time_series, positions):
        self.time_series = time_series
        self.positions = positions
        self.num_beads = positions.shape[1]
        if len(time_series) > 1:
            self.dt = time_series[1] - time_series[0]
        else:
            self.dt = None

    def _get_zero_crossing_times(self, bead_index, dimension=0):
        signal = self.positions[:, bead_index, dimension]
        tvals = self.time_series
        crossing_times = []

        # Use mean value instead of zero
        mean_val = np.mean(signal)

        for i in range(1, len(signal)):
            # Check for crossings around the mean
            if (signal[i - 1] <= mean_val) and (signal[i] > mean_val):
                denom = signal[i] - signal[i - 1]
                frac = 0.0
                if abs(denom) > 1e-14:
                    frac = (mean_val - signal[i - 1]) / denom
                crossing_time = tvals[i - 1] + frac * (tvals[i] - tvals[i - 1])
                crossing_times.append(crossing_time)

        return crossing_times

    def count_crossings(self, bead_index, dimension=0):
        """
        Count the total number of times a bead crosses its mean position
        (both upward and downward).
        """
        signal = self.positions[:, bead_index, dimension]
        mean_val = np.mean(signal)
        
        # Binary signal: 1 if above mean, 0 if below
        above = (signal > mean_val).astype(int)
        
        # Crossings occur when the difference between adjacent elements is non-zero
        crossings = np.sum(np.abs(np.diff(above)))
        return crossings

    def _get_mean_period(self, bead_index=0, dimension=0):
        crossing_times = self._get_zero_crossing_times(bead_index, dimension)
        if len(crossing_times) < 2:
            return np.nan
        intervals = np.diff(crossing_times)
        return np.mean(intervals)

    def _get_phase_differences(self, ref_bead=0, bead_j=1, dimension=0):
        ref_times = self._get_zero_crossing_times(ref_bead, dimension)
        j_times = self._get_zero_crossing_times(bead_j, dimension)
        min_len = min(len(ref_times), len(j_times))
        if min_len < 2:
            return np.array([]), np.array([])
        ref_times = ref_times[:min_len]
        j_times = j_times[:min_len]
        T0 = self._get_mean_period(ref_bead, dimension)
        T_mean = self._get_mean_period(ref_bead, dimension)
        if np.isnan(T_mean):
            print(f"Warning: Mean period for bead {ref_bead} is NaN!")
        delta_ts = np.array(j_times) - np.array(ref_times)
        delta_phis = delta_ts / T0

        return delta_ts, delta_phis

    def coherence_windowed_real_sum(self, ref_bead=0, dimension=0, periods_per_window=10):
        """
        Windowed phase-locking on a 0–1 scale via std of the unit-phasor.

        Returns
        -------
        numpy.ndarray
            Length num_beads; element j is the average
            coherence C ∈ [0,1] for bead j vs ref_bead (NaN if not enough crossings).
            C=1 → perfect phase lock; C=0 → no locking (uniform phases).
        """
        coherence_array = np.zeros(self.num_beads)

        for j in range(self.num_beads):
            if j == ref_bead:
                coherence_array[j] = 1.0
                continue

            _, delta_phis = self._get_phase_differences(ref_bead, j, dimension)
            n = len(delta_phis)
            if n < periods_per_window:
                coherence_array[j] = np.nan
                continue

            window_cohs = []
            for start in range(0, n, periods_per_window):
                end = start + periods_per_window
                if end > n:
                    break

                window = delta_phis[start:end]
                z = np.exp(1j * window)                   # unit phasors
                mu = np.mean(z)                           # mean phasor
                std_complex = np.sqrt(np.mean(np.abs(z - mu)**2))
                coh = 1.0 - std_complex                    # maps [0,1]→[1,0]
                window_cohs.append(coh)

            if not window_cohs:
                coherence_array[j] = np.nan
            else:
                # drop first 10% of windows then average
                drop = int(len(window_cohs) * 0.1)
                coherence_array[j] = np.mean(window_cohs[drop:])

        return coherence_array
