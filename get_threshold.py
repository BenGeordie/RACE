import tensorflow as tf
from race import Race
from lsh_functions import PStableHash
from utils import load_avazu_csv, load_criteo_csv, load_movielens_csv
from models import make_avazu_embedding_model, make_criteo_embedding_model, make_movielens_embedding_model
import numpy as np

def make_default_race(embedding_model):
    repetitions = 100
    concatenations = 1
    buckets = 10_000
    p = 2.0
    seed = 314150
    hash_module = PStableHash(embedding_model.output_shape[1], num_hashes=repetitions * concatenations, p=p, seed=seed)
    return Race(repetitions, concatenations, buckets, hash_module)

def compute_scores(dataset: tf.data.Dataset, save_path: str, embedding_model: tf.Module):
    race = make_default_race(embedding_model)
    train_data = dataset.batch(512)
    scored_iterable = train_data.map(lambda x, y: (race.score(embedding_model(x))[0], y))
    
    all_scores = np.ndarray(shape=(0,2))
    last_i = 0
    last_saved = 0
    intervals = 1000
    for i, (scores, y) in enumerate(scored_iterable):
        print(i, end='\r', flush=True)
        if i <= 32000:
            continue
        all_scores = np.concatenate([all_scores, np.stack([scores, y], axis=1)])
        if i - last_saved >= intervals:
            np.save(save_path + "_" + str(i // intervals), all_scores)
            all_scores = np.ndarray(shape=(0,2))
            last_saved = i
        if i - last_i >= 1000:
            print("", flush=True)
            last_i = i
    print("\nDone")
    
    np.save(save_path + "_last", all_scores)

# criteo_base = "AAAAA"
# criteo_train = load_criteo_csv(criteo_base + "train_.csv")
# criteo_embed = make_criteo_embedding_model()
# compute_scores(criteo_train, "")

# print("Avazu!")
# avazu_base = "/Users/benitogeordie/Downloads/Avazu_x4/"
# avazu_train = load_avazu_csv(avazu_base + "train_contig_noid.csv")
# avazu_embed = make_avazu_embedding_model()
# compute_scores(avazu_train, avazu_base + "scores", avazu_embed)

# print("Movielens!")
# movielens_base = "/Users/benitogeordie/Downloads/Movielenslatest_x1/"
# movielens_train = load_movielens_csv(movielens_base + "train_contig.csv")
# movielens_embed = make_movielens_embedding_model()
# compute_scores(movielens_train, movielens_base + "scores", movielens_embed)


def get_sampling_rate(scores, keep_first_n, threshold, accept_prob):
    after_n = scores[keep_first_n:, :]

    majority_after_n = after_n[after_n[:, 1] == 0]
    num_majority_after_n = majority_after_n.shape[0]
    num_minority_after_n = after_n.shape[0] - num_majority_after_n
    num_passing_majority_after_n = majority_after_n[majority_after_n[:, 0] < threshold].shape[0]
    num_lucky_majority_after_n = (num_majority_after_n - num_passing_majority_after_n) * accept_prob
    total = scores.shape[0]
    return (keep_first_n + num_lucky_majority_after_n + num_passing_majority_after_n + num_minority_after_n) / total


import matplotlib.pyplot as plt

movielens_savefiles = ['/Users/benitogeordie/Downloads/Movielenslatest_x1/scores_last.npy']
scores = np.concatenate([np.load(f) for f in movielens_savefiles])

avazu_savefiles = [f'/Users/benitogeordie/Downloads/Avazu_x4/scores_{i}.npy' for i in range(1, 64)] + ['/Users/benitogeordie/Downloads/Avazu_x4/scores_last.npy']
scores = np.concatenate([np.load(f) for f in avazu_savefiles])

# hist = np.histogram(scores[:,0])
# plt.hist(scores[:,0], bins=100)
# plt.show()

# movielens_rates = [get_sampling_rate(scores, keep_first_n=10_000, threshold=t, accept_prob=0.1) for t in np.arange(0.04, 0.065, 0.002)]
avazu_rates = [get_sampling_rate(scores, keep_first_n=10_000, threshold=t, accept_prob=0.1) for t in np.arange(0.01, 0.015, 0.0005)]

rates = avazu_rates
plt.plot(rates)
plt.show()

# print(get_sampling_rate(scores, keep_first_n=10_000, threshold=0.05, accept_prob=0.1))


#TODO: Range of taus for movielens: np.arange(0.04, 0.065, 0.002)
#TODO: Range of taus for avazu: np.arange(0.01, 0.015, 0.0005)