import pickle
import numpy as np
import os

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        data = dataset[b'data']  # shape: (10000, 3072)
        labels = dataset[b'labels']
        data = data.reshape(-1, 3, 32, 32)  # 如果想保持channel-first
        labels = np.array(labels)
    return data, labels


def load_dataset(root_dir):
    d_list = []
    l_list = []
    for i in range(1, 6):
        f = os.path.join(root_dir, f'data_batch_{i}')
        data, labels = load_batch(f)
        d_list.append(data)
        l_list.append(labels)
    data_train = np.concatenate(d_list)
    labels_train = np.concatenate(l_list)
    
    data_test, labels_test = load_batch(os.path.join(root_dir, 'test_batch'))
    # reshape => (N, 3072)
    data_train = data_train.reshape(data_train.shape[0], -1)  # (N, 3072)
    data_test = data_test.reshape(data_test.shape[0], -1)  # (N, 3072)

    return data_train, labels_train, data_test, labels_test

def split_train_valid_set(data, labels, val_ratio=0.2, random_seed=2025):
    np.random.seed(random_seed) 
    num_data = data.shape[0]
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    split = int(num_data * (1 - val_ratio))
    
    train_idx, val_idx = indices[:split], indices[split:]
    data_train, labels_train = data[train_idx], labels[train_idx]
    data_val, labels_val = data[val_idx], labels[val_idx]
    
    return data_train, labels_train, data_val, labels_val


class Dataloader:
    def __init__(self, data, labels, batch_size=128, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = data.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self._reset()

    def _reset(self):
        self.current_idx = 0
        if self.shuffle:
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            self._reset()
            raise StopIteration
        start = self.current_idx
        end = min(start + self.batch_size, self.num_samples)
        batch_data = self.data[start:end]
        batch_labels = self.labels[start:end]
        self.current_idx = end
        return batch_data, batch_labels

if __name__ == '__main__':
    root_path = './dataset'
    data_train, labels_train, data_test, labels_test = load_dataset(root_path)
    data_train, labels_train, data_val, labels_val = split_train_valid_set(data_train, labels_train, 0.2)
    train = Dataloader(data_train, labels_train, batch_size=128, shuffle=True)
    val = Dataloader(data_val, labels_val, batch_size=128, shuffle=True)
    test = Dataloader(data_test, labels_test, batch_size=128, shuffle=False)
    for batch_data, batch_labels in train:
        print(batch_data.shape, batch_labels.shape)
        break  