import os
import numpy as np

def load_har_signals(data_path, group):

    # channels from UCI HAR dataset
    channels = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    all_signals = []

    for channel_name in channels:
        file_name = f"{channel_name}_{group}.txt"

        file_path = os.path.join(data_path, group, "Inertial Signals", file_name)
        current_signal = np.loadtxt(file_path)
        all_signals.append(current_signal)

    # stack along channel dimension
    X = np.stack(all_signals, axis=1)

    return X


def load_har_labels(data_path, group):
    label_path = os.path.join(data_path, group, f"y_{group}.txt")
    labels = np.loadtxt(label_path, dtype=np.int64)
    return labels - 1


def get_uci_har_data(base_path="UCI HAR Dataset"):

    # loads and normalises
    X_train = load_har_signals(base_path, "train")
    y_train = load_har_labels(base_path, "train")
    X_test = load_har_signals(base_path, "test")
    y_test = load_har_labels(base_path, "test")

    train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    train_std = np.std(X_train, axis=(0, 2),keepdims=True)

    epsilon = 1e-8

    X_train = (X_train - train_mean) / (train_std + epsilon)
    X_test = (X_test - train_mean) / (train_std + epsilon)
    # print(np.unique(y_train))

    return (X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_uci_har_data()
    print("\nDataset loaded successfully.")