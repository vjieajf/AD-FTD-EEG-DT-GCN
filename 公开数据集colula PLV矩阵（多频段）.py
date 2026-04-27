import os

import mne
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, resample
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_backend

# Main dataset directory
main_dir = 'E:/静息态脑电数据/111/ds004504/derivatives/'

# Five standard frequency bands for multiband output
frequency_bands = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]

# Keep raw objects by subject id
raw_data_dict = {}


def gumbel_copula(u, v, theta):
    u = np.clip(u, 1e-8, 1 - 1e-8)
    v = np.clip(v, 1e-8, 1 - 1e-8)
    return np.exp(-(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1.0 / theta)))


def copula_log_likelihood(theta, u, v, copula_func, epsilon=1e-8):
    theta_val = float(np.atleast_1d(theta)[0])
    copula_vals = copula_func(u, v, theta_val)
    return -np.sum(np.log(np.clip(copula_vals, epsilon, None)))


def compute_copula_matrix(epoch_data, copula_func, initial_theta=1.0, bounds=None):
    """Compute channel-wise Copula FC matrix for one epoch (channels x samples)."""
    epoch_data = np.asarray(epoch_data, dtype=float)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(epoch_data.T).T

    n_channels = standardized_data.shape[0]
    copula_mat = np.full((n_channels, n_channels), np.nan, dtype=float)

    for target in range(n_channels):
        target_signal = standardized_data[target]
        for source in range(n_channels):
            if target == source:
                continue

            source_signal = standardized_data[source]
            u = (rankdata(target_signal) + 0.5) / (len(target_signal) + 1)
            v = (rankdata(source_signal) + 0.5) / (len(source_signal) + 1)

            try:
                result = minimize(
                    copula_log_likelihood,
                    x0=np.array([initial_theta], dtype=float),
                    args=(u, v, copula_func),
                    bounds=bounds,
                    method='L-BFGS-B',
                )
                optimal_theta = float(result.x[0])
                copula_vals = copula_func(u, v, optimal_theta)
                copula_mat[source, target] = np.nanmean(copula_vals)
            except Exception as exc:
                print(f"Copula estimation failed at ({source}, {target}): {exc}")

    np.fill_diagonal(copula_mat, 0.0)
    copula_mat = np.nan_to_num(copula_mat, nan=0.0, posinf=0.0, neginf=0.0)
    copula_mat = np.clip(copula_mat, 0.0, 0.5)
    return copula_mat


def process_subject(sub_id):
    """单个被试的完整处理流程，供并行调用。"""
    sub_dir = f'sub-{sub_id:03d}'
    file_path = os.path.join(main_dir, sub_dir, 'eeg', f'{sub_dir}_task-eyesclosed_eeg.set')

    if not os.path.exists(file_path):
        print(f"File for {sub_dir} not found.")
        return sub_id - 1, np.zeros((19, 19 * len(frequency_bands)), dtype=float)

    raw_data = mne.io.read_raw_eeglab(file_path, preload=True)
    boundary_events = raw_data.annotations.description == 'boundary'

    if np.any(boundary_events):
        indices_to_delete = [
            idx for idx, desc in enumerate(raw_data.annotations.description) if desc == 'boundary'
        ]
        raw_data.annotations.delete(indices_to_delete)
        print(f"Deleted boundary events for {sub_dir}")
    else:
        print(f"No boundary events found for {sub_dir}")

    data = raw_data.get_data()
    print(f"Processing {sub_dir}, raw shape={data.shape}")

    original_sampling_rate = float(raw_data.info['sfreq'])
    target_sampling_rate = 250
    if int(round(original_sampling_rate)) != target_sampling_rate:
        new_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
        data = resample(data, new_samples, axis=1)

    fs = target_sampling_rate
    n_samples_per_epoch = int(5000 * fs / 1000)  # 5s per epoch
    filter_order = 4

    subject_band_matrices = []

    for band_name, low, high in frequency_bands:
        nyq = 0.5 * fs
        low_cut = low / nyq
        high_cut = high / nyq
        bandpass_coeffs = butter(filter_order, [low_cut, high_cut], btype='band', output='ba')
        b, a = bandpass_coeffs[0], bandpass_coeffs[1]

        filtered_eeg = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_eeg[ch, :] = filtfilt(b, a, data[ch, :])

        n_epochs = filtered_eeg.shape[1] // n_samples_per_epoch
        if n_epochs == 0:
            print(f"Warning: {sub_dir} {band_name} has no valid 5s epoch, use zeros.")
            subject_band_matrices.append(np.zeros((19, 19), dtype=float))
            continue

        # Scheme B: concatenate all 5-second epochs first, then estimate Copula once per band.
        # This is a deliberate simplified ablation for much faster runtime.
        band_epoch_segments = []
        for epoch_idx in range(n_epochs):
            start = epoch_idx * n_samples_per_epoch
            end = (epoch_idx + 1) * n_samples_per_epoch
            band_epoch_segments.append(filtered_eeg[0:19, start:end])

        concatenated_band_data = np.concatenate(band_epoch_segments, axis=1)
        band_copula_matrix = compute_copula_matrix(
            concatenated_band_data,
            gumbel_copula,
            initial_theta=1.0,
            bounds=[(1.0, 10.0)],
        )

        if np.isfinite(band_copula_matrix).any():
            subject_band_matrices.append(band_copula_matrix)
        else:
            print(f"Warning: {sub_dir} {band_name} has no valid copula result, use zeros.")
            subject_band_matrices.append(np.zeros((19, 19), dtype=float))

    subject_multiband_3d = np.stack(subject_band_matrices, axis=-1)
    subject_multiband_2d = subject_multiband_3d.reshape(19, 19 * len(frequency_bands))
    return sub_id - 1, subject_multiband_2d


def main():
    # Fixed output container: (88, 19, 95)
    all_subject_copula_multiband = np.zeros((88, 19, 19 * len(frequency_bands)), dtype=float)

    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_subject)(sub_id) for sub_id in range(1, 89)
        )

    for sub_idx, subject_matrix in results:
        all_subject_copula_multiband[sub_idx] = subject_matrix

    # Save unified outputs (.txt and .npy)
    save_dir = 'H:/数据/Daima/pythonProject/保存数据'
    os.makedirs(save_dir, exist_ok=True)

    txt_filename = os.path.join(save_dir, 'avg_copula_matrix_multiband.txt')
    with open(txt_filename, 'w', encoding='utf-8') as file:
        for matrix in all_subject_copula_multiband:
            for row in matrix:
                file.write('\t'.join(f'{value:.6f}' for value in row))
                file.write('\n')
            file.write('\n')

    npy_filename = os.path.join(save_dir, 'avg_copula_matrix_multiband.npy')
    np.save(npy_filename, all_subject_copula_multiband)

    print(f"Saved multiband Copula txt: {txt_filename}")
    print(f"Saved multiband Copula npy: {npy_filename}, shape={all_subject_copula_multiband.shape}")


if __name__ == '__main__':
    main()


