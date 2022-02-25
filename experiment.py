from this import d
from typing import List
from race import Race
from data_loader import DataLoader
from criteo_preprocess import quantize
from criteo_preprocess import separate_domains_by_range
from model import make_criteo_model
import numpy as np
import tensorflow as tf
import random
import time

def process_x_train_neg(x_train_neg):
    # Based on the ranges of the continuous fields (split into approximately 10 bins if range allows)
    bin_configs = [
        (580, 1),
        (26_000, 1),
        (6600, 1),
        (1_000_000, 1),
        (5640, 1),
        (605, 1),
        (2902, 1),
        (4, 1),
        (30, 1),
        (401, 1),
        (740, 1),
    ]

    quantized = np.concatenate([quantize(x_train_neg[:,i:i+1], bin_width, n_bins) for i, (bin_width, n_bins) in enumerate(bin_configs)] + [x_train_neg[:,13:]], axis=1)
    quantized = separate_domains_by_range(quantized)
    quantized = tf.data.Dataset.from_tensor_slices((quantized, np.zeros(quantized.shape[0])))

def race_map(race: Race, threshold: float, accept_prob: float, seed: int):
    @tf.function
    def fn(x):
        score = race.index_then_query(x)
        if score > threshold:
            return x
        state = random.getstate()
        random.seed(seed)
        if random.random() < accept_prob:
            scaled = x / accept_prob
        else:
            scaled = x * 0
        random.setstate(state)
        return scaled
    return fn

def log_history(log_file, history):
    log_file.write(history + '\n')

def run_experiment(
    input_path: str, 
    repetitions: List[int], 
    concatenations: List[int], 
    buckets: List[int], 
    thresholds: List[float],
    accept_probs: List[float],
    epochs: int,
    log_file_path: str,
    seed: int):

    with open(log_file_path) as log:
        # Time per epoch
        # Accuracy per epoch
        # Parameters

        data = DataLoader(input_path)
        test_data = data.get_test()
        x_train_neg = data.get_x_train_neg()
        x_train_neg = process_x_train_neg(x_train_neg)

        for rep in repetitions:
            for conc in concatenations:
                for buck in buckets:
                    for thresh in thresholds:
                        for prob in accept_probs:
                            race = Race(rep, conc, buck, seed)
                            # Do a map first to get the weights
                            x_train_neg_filtered = x_train_neg.map(race_map(race, thresh, prob, seed)).filter(tf.math.count_nonzero)
                            train_data = data.get_train(x_train_neg_filtered)
                            #  TODO: THIS DOESNT WORK YET SINCE X_TRAIN_NEG_FILTERED IS QUANTIZED AND X_TRAIN_POS IS NOT
                            
                            model = make_criteo_model()
                            
                            entry = {
                                "parameters": {
                                    "rep": rep,
                                    "conc": conc,
                                    "buck": buck,
                                    "thresh": thresh,
                                    "prob": prob
                                },
                                "epoch_train_times": [],
                                "epoch_test_times": [],
                                "epoch_train_accuracies": [],
                                "epoch_test_accuracies": [],
                            }
                            for _ in epochs:
                                start_train = time.time()
                                h = model.fit(train_data)
                                end_train = time.time()
                                entry["epoch_train_times"].append(end_train - start_train)
                                entry["epoch_train_accuracies"].append(h.history["accuracy"][-1])

                                start_test = time.time()
                                _, acc = model.evaluate(test_data)
                                end_test = time.time()
                                entry["epoch_test_times"].append(end_test - start_test)
                                entry["epoch_test_accuracies"].append(acc)
                            log_history(log, entry)

                                

                                

                        


    

