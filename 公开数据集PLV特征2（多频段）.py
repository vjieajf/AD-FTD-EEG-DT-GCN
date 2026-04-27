import os

import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample

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


def extract_phases(data):
    """data shape: (channels, samples)."""
    analytic_signal = hilbert(data, axis=1)
    return np.angle(analytic_signal)


def compute_plv_matrix(epoch_data):
    """Compute PLV matrix for one epoch, output shape: (channels, channels)."""
    phases = extract_phases(epoch_data)
    n_channels = phases.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels), dtype=float)

    for i in range(n_channels):
        plv_matrix[i, i] = 1.0
        for j in range(i + 1, n_channels):
            phase_diff = phases[i, :] - phases[j, :]
            plv_val = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv_val
            plv_matrix[j, i] = plv_val

    return plv_matrix


def compute_pearson_matrix(epoch_data):
    """Compute Pearson correlation matrix for one epoch."""
    pearson_matrix = np.corrcoef(epoch_data)
    pearson_matrix = np.nan_to_num(pearson_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(pearson_matrix, 1.0)
    return pearson_matrix


# Read subjects and remove boundary annotations
for sub_id in range(1, 89):
    sub_dir = f'sub-{sub_id:03d}'
    file_path = os.path.join(main_dir, sub_dir, 'eeg', f'{sub_dir}_task-eyesclosed_eeg.set')

    if os.path.exists(file_path):
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

        print(f"Loaded {sub_dir}: {raw_data.info['nchan']} channels, {raw_data.n_times} samples")
        raw_data_dict[sub_dir] = raw_data
    else:
        print(f"File for {sub_dir} not found.")


# Fixed output containers: (88, 19, 95)
all_subject_plv_multiband = np.zeros((88, 19, 19 * len(frequency_bands)), dtype=float)
all_subject_pearson_multiband = np.zeros((88, 19, 19 * len(frequency_bands)), dtype=float)

for sub_idx, sub_id in enumerate(range(1, 89)):
    sub_dir = f'sub-{sub_id:03d}'
    sub_data = raw_data_dict.get(sub_dir)

    if sub_data is None:
        print(f"Warning: {sub_dir} data missing, keep zeros.")
        continue

    data = sub_data.get_data()
    print(f"Processing {sub_dir} ({sub_idx + 1}/88), raw shape={data.shape}")

    # Resample to 250 Hz
    original_sampling_rate = float(sub_data.info['sfreq'])
    target_sampling_rate = 250
    if int(round(original_sampling_rate)) != target_sampling_rate:
        new_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
        data = resample(data, new_samples, axis=1)

    fs = target_sampling_rate
    n_samples_per_epoch = int(5000 * fs / 1000)  # 5s per epoch
    filter_order = 4

    subject_band_plv = []
    subject_band_pearson = []

    for band_name, low, high in frequency_bands:
        nyq = 0.5 * fs
        low_cut = low / nyq
        high_cut = high / nyq
        b, a = butter(filter_order, [low_cut, high_cut], btype='band')

        filtered_eeg = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_eeg[ch, :] = filtfilt(b, a, data[ch, :])

        n_epochs = filtered_eeg.shape[1] // n_samples_per_epoch
        if n_epochs == 0:
            print(f"Warning: {sub_dir} {band_name} has no valid 5s epoch, use zeros.")
            subject_band_plv.append(np.zeros((19, 19), dtype=float))
            subject_band_pearson.append(np.zeros((19, 19), dtype=float))
            continue

        plv_sum = np.zeros((19, 19), dtype=float)
        pearson_sum = np.zeros((19, 19), dtype=float)
        valid_epoch_count = 0

        for epoch_idx in range(n_epochs):
            start = epoch_idx * n_samples_per_epoch
            end = (epoch_idx + 1) * n_samples_per_epoch
            epoch_data = filtered_eeg[0:19, start:end]

            plv_matrix = compute_plv_matrix(epoch_data)
            pearson_matrix = compute_pearson_matrix(epoch_data)

            plv_sum += np.nan_to_num(plv_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            pearson_sum += np.nan_to_num(pearson_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            valid_epoch_count += 1

        if valid_epoch_count == 0:
            subject_band_plv.append(np.zeros((19, 19), dtype=float))
            subject_band_pearson.append(np.zeros((19, 19), dtype=float))
        else:
            avg_plv = plv_sum / float(valid_epoch_count)
            avg_pearson = pearson_sum / float(valid_epoch_count)
            np.fill_diagonal(avg_plv, 1.0)
            np.fill_diagonal(avg_pearson, 1.0)
            subject_band_plv.append(avg_plv)
            subject_band_pearson.append(avg_pearson)

    # 19x19x5 -> 19x95
    subject_plv_3d = np.stack(subject_band_plv, axis=-1)
    subject_pearson_3d = np.stack(subject_band_pearson, axis=-1)

    all_subject_plv_multiband[sub_idx] = subject_plv_3d.reshape(19, 19 * len(frequency_bands))
    all_subject_pearson_multiband[sub_idx] = subject_pearson_3d.reshape(19, 19 * len(frequency_bands))


# Save unified outputs (.txt and .npy)
save_dir = 'H:/数据/Daima/pythonProject/保存数据'
os.makedirs(save_dir, exist_ok=True)

plv_txt = os.path.join(save_dir, 'avg_plv_matrix_multiband.txt')
with open(plv_txt, 'w', encoding='utf-8') as file:
    for matrix in all_subject_plv_multiband:
        for row in matrix:
            file.write('\t'.join(f'{value:.6f}' for value in row))
            file.write('\n')
        file.write('\n')

pearson_txt = os.path.join(save_dir, 'avg_pearson_matrix_multiband.txt')
with open(pearson_txt, 'w', encoding='utf-8') as file:
    for matrix in all_subject_pearson_multiband:
        for row in matrix:
            file.write('\t'.join(f'{value:.6f}' for value in row))
            file.write('\n')
        file.write('\n')

plv_npy = os.path.join(save_dir, 'avg_plv_matrix_multiband.npy')
pearson_npy = os.path.join(save_dir, 'avg_pearson_matrix_multiband.npy')
np.save(plv_npy, all_subject_plv_multiband)
np.save(pearson_npy, all_subject_pearson_multiband)

print(f"Saved PLV multiband txt: {plv_txt}")
print(f"Saved Pearson multiband txt: {pearson_txt}")
print(f"Saved PLV multiband npy: {plv_npy}, shape={all_subject_plv_multiband.shape}")
print(f"Saved Pearson multiband npy: {pearson_npy}, shape={all_subject_pearson_multiband.shape}")
