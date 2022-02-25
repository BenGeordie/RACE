import numpy as np
import tensorflow as tf

class DataLoader:
    def __init__(self, path: str):
        data = np.load(path)
        x_cat = data['X_cat'].astype(np.int32)
        x_int = data['X_int'].astype(np.int32)
        y = data['y']

        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)

        n_train = int(len(idxs)*0.8)
        
        train_idxs = idxs[:n_train]
        test_idxs = idxs[n_train:]

        self.y_test = y[test_idxs]
        y_train = y[train_idxs]

        pos_idxs = np.where(y_train == 1)[0]
        neg_idxs = np.where(y_train == 0)[0]

        self.x_test = np.concatenate((x_int[test_idxs], x_cat[test_idxs]), axis=1)
        self.x_train_neg = np.concatenate((x_int[train_idxs][neg_idxs], x_cat[train_idxs][neg_idxs]), axis=1)
        self.x_train_pos = np.concatenate((x_int[train_idxs][pos_idxs], x_cat[train_idxs][pos_idxs]), axis=1)

    def get_x_train_neg(self):
        return self.x_train_neg

    def get_train(self, x_train_neg_filtered: tf.data.Dataset):
        x_train = tf.data.Dataset.from_tensor_slices((self.x_train_pos, np.ones(self.x_train_pos.shape[0])))
        x_train = x_train.concatenate(x_train_neg_filtered)
        x_train = x_train.shuffle(1000)
        return x_train
    
    def get_test(self):
        return self.y_test, self.x_test
