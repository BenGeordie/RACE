from typing import List
import tensorflow as tf
from race import Race
import pdb

# Tested lightly
def load_csv(paths: List[str], sample_dim: int):
    """Loads CSV files and creates a dataset of (sample, target) tuples.
    Arguments:
        paths: List of strings. Paths to CSV files.
        sample_dim: integer. The dimension of each sample in the dataset.
    Returns: 
        A TensorFlow dataset of (sample, target) tuples.
    """
    data = tf.data.experimental.CsvDataset(
        paths, record_defaults=[tf.int64]+[tf.float32 for _ in range(sample_dim)],header=True)
    return data.map(lambda *line: (tf.stack(line[1:]), line[0]))

# Untested
def split_train_val_test(
    dataset: tf.data.Dataset, dataset_size: int, 
    train_split: float=0.8, val_split: float=0.1, test_split: float=0.1, 
    shuffle=True, shuffle_size=10000):
    """Splits dataset into train, validation, and test datasets.
    """

    assert (train_split + val_split + test_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        dataset = dataset.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_ds = dataset.take(train_size)    
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


# Untested
@tf.function
def weight_and_filter(dataset: tf.data.Dataset, weight_fn):
    """Weights each sample in the dataset using a weighting function, and filters out elements with 0 weight.
    Arguments:
        data: TensorFlow dataset of (sample, target) tuples.
        weight_fn: A function that takes in samples and weights them.
    Returns:
        A TensorFlow dataset of (sample, target, sample_weight) tuples.
    """
    def map_fn(x, y):
        weights = tf.ones(shape=tf.shape(y), dtype=tf.float32)
        weights = tf.math.multiply(weights,weight_fn(x,y))

        return (x, y, weights)
    dataset_map = dataset.map(map_fn)
    dataset_map_unbatch = dataset_map.unbatch()
   
    return dataset_map_unbatch.filter(lambda _x, _y, w: w > 0) 
# Untested
def weight_with_race(race: Race, embedding_model: tf.Module, accept_first_n: int, score_threshold: float, accept_prob: float):
    """Function factory for weighting samples with RACE. To be used in conjunction with weight_and_filter, defined above.
    Note that all positive samples will be weighted 1.0.
    For convenience, further references to "samples" refer to negative samples, unless otherwise specified.
    Arguments: 
        race: Initialized RACE data structure.
        embedding_model: TensorFlow model for embedding samples into a RACE-compatible space.
        accept_first_n: integer. Number of samples to be accepted and weighted 1.0 before 
            reweighting based on RACE and random sampling.
        score_threshold: float. Samples with lower RACE scores than this threshold are weighted 1.0.
        accept_prob: float. If we have seen more than accept_first_n samples, we accept the sample and 
            weight it 1 / accept_prob.
    """
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])
    
    def _weight(x, y):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """
#         if len(tf.shape(x)) < 2:
#             x = tf.reshape(x, (1, tf.shape(x)[0]))
        
        e = embedding_model(x)

        # Accept everything if we have not seen too many samples.
        if race.samples_seen() < tf.constant(accept_first_n, dtype=tf.int64):
            return tf.ones(shape=tf.shape(e)[0])
        
        # score_threshold <= 1.0
        # score_threshold - scores is in the range [-1 + score_threshold, score_threshold]
        # Thus, taking the ceiling results in 1 if score < threshold, 0 otherwise.
        scores = race.score(e)
        print(scores)
        accepted_by_score_weights = tf.cast(tf.math.ceil(score_threshold - scores), dtype=tf.float32)
        
        # Accepted by chance if random_num < accept_prob.
        # As above, taking the ceiling results in 1 if random_num < accept_prob, 0 otherwise.
        random_num = tf.random.uniform(shape=[tf.shape(e)[0]])
        accepted_by_chance = tf.math.ceil(tf.constant(accept_prob) - random_num)
        # If accepted, weight is 1 / accept_prob.
        accepted_by_chance_weights = tf.where(tf.cast(accepted_by_chance, dtype=tf.bool), tf.math.reciprocal_no_nan(accept_prob), [0.0])

        # If passing accepted_by_score_weights == 1.0, keep the weight. Otherwise, if accepted by random chance,
        # weight by 1 / accept_prob
        sampled_negative_weights = tf.where(tf.cast(accepted_by_score_weights, dtype=tf.bool), accepted_by_score_weights, accepted_by_chance_weights)
        
        # Accept if positive, weight accordingly otherwise.
        return tf.where(tf.cast(y, dtype=tf.bool), [1.0], sampled_negative_weights)


    return _weight

