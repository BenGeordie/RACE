from typing import List
import tensorflow as tf
from race import Race
import pdb

# Tested lightly
def load_criteo_csv(paths: List[str]):
    """Loads criteo CSV and creates a dataset of (sample, target) tuples.

    Criteo data has label in first column, followed by 39 continuous and categorical features.

    Arguments:
        paths: List of strings. Paths to CSV files.
        sample_dim: integer. The dimension of each sample in the dataset.
    Returns: 
        A TensorFlow dataset of (sample, target) tuples.
    """
    data = tf.data.experimental.CsvDataset(
        paths, record_defaults=[tf.int64]+[tf.float32 for _ in range(39)],header=True)
    return data.map(lambda *line: (tf.stack(line[1:]), line[0]))

def load_csv_with_score(paths: List[str]):
    """Loads CSV files and creates a dataset of (sample, target, scores) tuples.
    Arguments:
        paths: List of strings. Paths to CSV files.
        sample_dim: integer. The dimension of each sample in the dataset.
    Returns:
        A TensorFlow dataset of (sample, target, scores) tuples.
    """
    data = tf.data.experimental.CsvDataset(
        paths, record_defaults=[tf.int64]+[tf.float32 for _ in range(40)],header=True)
    return data.map(lambda *line: (tf.stack(line[1:-1]), line[0], line[-1]))


def load_avazu_csv_with_score(paths: List[str]):
    """Loads CSV files and creates a dataset of (sample, target, scores) tuples.
    Arguments:
        paths: List of strings. Paths to CSV files.
        sample_dim: integer. The dimension of each sample in the dataset.
    Returns:
        A TensorFlow dataset of (sample, target, scores) tuples.
    """
    data = tf.data.experimental.CsvDataset(
        paths, record_defaults=[tf.int64]+[tf.float32 for _ in range(23)],header=True)
    return data.map(lambda *line: (tf.stack(line[1:-1]), line[0], line[-1]))


def load_avazu_csv(paths: List[str]):
    """Loads avazu CSV and creates a dataset of (sample, target) tuples.

    Avazu data has id in first column, label in second column, followed by 22 categorical features,
    but we preprocessed the dataset to discard the first column.

    Arguments:
        paths: List of strings. Paths to CSV files.
        sample_dim: integer. The dimension of each sample in the dataset.
    Returns: 
        A TensorFlow dataset of (sample, target) tuples.
    """
    data = tf.data.experimental.CsvDataset(
        paths, record_defaults=[tf.int64]+[tf.float32 for _ in range(22)],header=True)
    return data.map(lambda *line: (tf.stack(line[1:]), line[0]))

def load_movielens_csv(paths: List[str]):
    """Loads movielens CSV and creates a dataset of (sample, target) tuples.

    Movielens data has label in first column, followed by 3 columns of categorical features.

    Arguments:
        paths: List of strings. Paths to CSV files.
        sample_dim: integer. The dimension of each sample in the dataset.
    Returns: 
        A TensorFlow dataset of (sample, target) tuples.
    """
    data = tf.data.experimental.CsvDataset(
        paths, record_defaults=[tf.int64]+[tf.float32 for _ in range(3)],header=True)
    return data.map(lambda *line: (tf.stack(line[1:]), line[0]))


# Untested
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
    return dataset_map_unbatch.filter(lambda _x, _y, w: w > 0).batch(512) # to do: pass batch_size to avoid hard coding 
    #return dataset_map # to save scores/weights offline
    
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
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _sample_after_first_n(scores, y):
        # score_threshold <= 1.0
        # score_threshold - scores is in the range [-1 + score_threshold, score_threshold]
        # Thus, taking the ceiling results in 1 if score < threshold, 0 otherwise.
        print(scores)
        accepted_by_score_weights = tf.cast(tf.math.ceil(score_threshold - scores), dtype=tf.float32)
        
        # Accepted by chance if random_num < accept_prob.
        # As above, taking the ceiling results in 1 if random_num < accept_prob, 0 otherwise.
        random_num = tf.random.uniform(shape=[tf.shape(scores)[0]])
        accepted_by_chance = tf.math.ceil(tf.constant(accept_prob) - random_num)
        # If accepted, weight is 1 / accept_prob.
        accepted_by_chance_weights = tf.where(tf.cast(accepted_by_chance, dtype=tf.bool), tf.math.reciprocal_no_nan(accept_prob), [0.0])

        # If passing accepted_by_score_weights == 1.0, keep the weight. Otherwise, if accepted by random chance,
        # weight by 1 / accept_prob
        sampled_negative_weights = tf.where(tf.cast(accepted_by_score_weights, dtype=tf.bool), accepted_by_score_weights, accepted_by_chance_weights)
        
        # Accept if positive, weight accordingly otherwise.
        return tf.where(tf.cast(y, dtype=tf.bool), [1.0], sampled_negative_weights)
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _weight(x, y):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """

        e = embedding_model(x)

        scores, n = race.score(e)

        return tf.cond(
            tf.less(n, tf.constant(accept_first_n, dtype=tf.int64)),
            lambda: tf.ones(shape=tf.shape(e)[0]),
            lambda: _sample_after_first_n(scores, y)
        )

    return _weight


def weight_and_filter_withscore(dataset: tf.data.Dataset, weight_fn):
    """Weights each sample in the dataset using a weighting function, and filters out elements with 0 weight.
    Arguments:
        data: TensorFlow dataset of (sample, target) tuples.
        weight_fn: A function that takes in samples and weights them.
    Returns:
        A TensorFlow dataset of (sample, target, sample_weight) tuples.
    """
    def map_fn(x, y, score):
        weights = tf.ones(shape=tf.shape(y), dtype=tf.float32)
        weights = tf.math.multiply(weights,weight_fn(x,y,score))

        return (x, y, score ,weights) # if you want to return score as well
#        return (x, y, weights)
    dataset_map = dataset.map(map_fn)
    dataset_map_unbatch = dataset_map.unbatch()
    return dataset_map_unbatch.filter(lambda _x, _y,_score, w: w > 0).batch(512) # To include scores column in the filtered dataset as well
  #  return dataset_map_unbatch.filter(lambda _x, _y, w: w >= 0).batch(512)


def weight_with_logloss_score(score_threshold: float, accept_prob: float):
    """Function factory for weighting samples with RACE. To be used in conjunction with weight_and_filter, defined above.
    Note that all positive samples will be weighted 1.0.
    For convenience, further references to "samples" refer to negative samples, unless otherwise specified.
    Arguments:
        race: Initialized RACE data structure.
        embedding_model: TensorFlow model for embedding samples into a RACE-compatible space.
        accept_first_n: integer. Number of samples to be accepted and weighted 1.0 before
            reweighting based on RACE and random sampling.
        score_threshold: float. Samples with lower RACE scores than this threshold are weighted 1.0.
        accept_prob: float. If we have seen more than accept_first_n samples,we accept the sample and
            weight it 1 / accept_prob.
    """
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64), tf.TensorSpec(shape=None, dtype=tf.float32)])
    
    def _weight(x, y, scores):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """

        # score_threshold <= 1.0
        #This part is the other way around the RACE scores. We want to downsample the points with lower logloss (score) than the threshhol and keep the ones with higher logloss score than the threshold. 
        # Thus, taking the ceiling results is 1 if score > threshold, 0 otherwise.

        accepted_by_score_weights = tf.cast(tf.math.ceil(scores - score_threshold), dtype=tf.float32)

       # accepted_by_score_weights = tf.cast(tf.math.ceil(score_threshold - tf.math.reciprocal(scores)), dtype=tf.float32) 
        # Accepted by chance if random_num < accept_prob.
        # As above, taking the ceiling results is 1 if random_num < accept_prob, 0 otherwise.
        random_num = tf.random.uniform(shape=[tf.shape(x)[0]])
        accepted_by_chance = tf.math.ceil(tf.constant(accept_prob) - random_num)
        # If accepted, weight is 1 / accept_prob.
        accepted_by_chance_weights = tf.where(tf.cast(accepted_by_chance, dtype=tf.bool), tf.math.reciprocal_no_nan(accept_prob), [0.0])

        # If passing accepted_by_score_weights == 1.0, keep the weight. Otherwise, if accepted by random chance,
        # weight by 1 / accept_prob
        sampled_negative_weights = tf.where(tf.cast(accepted_by_score_weights, dtype=tf.bool), accepted_by_score_weights, accepted_by_chance_weights)

        # Accept if positive, weight accordingly otherwise.
        return tf.where(tf.cast(y, dtype=tf.bool), [1.0], sampled_negative_weights)


    return _weight
                  
    
    
def weight_random(embedding_model: tf.Module, down_sampling_rate: float):
    """
    """
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])

    def _weight(x, y):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """
        accept_prob = 1 - down_sampling_rate

        # Accepted by chance if random_num < accept_prob.
        # As above, taking the ceiling results in 1 if random_num < accept_prob, 0 otherwise.
        random_num = tf.random.uniform(shape=[tf.shape(x)[0]])
        accepted_by_chance = tf.math.ceil(tf.constant(accept_prob) - random_num)
        sampled_negative_weights = tf.where(tf.cast(accepted_by_chance, dtype=tf.bool), [1.0], [0.0])

        # Accept if positive, weight accordingly otherwise.
        return tf.where(tf.cast(y, dtype=tf.bool), [1.0], sampled_negative_weights)


    return _weight
