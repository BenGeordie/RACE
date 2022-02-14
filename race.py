import tensorflow as tf
import random
import sys
import numpy as np

@tf.function
def minhash(x: tf.Tensor, n_hashes: int, hash_range: int, seed: int):
    """
    Assumes x is a one-dimensional tensor with dtype tf.int64
    """
    state = random.getstate()
    random.seed(seed)
    hashes = [
        (x * random.randint(0, sys.maxsize) + random.randint(0, sys.maxsize)) % hash_range
        for _ in range(n_hashes)]
    minhashes = tf.stack([tf.reduce_min(h) for h in hashes])
    random.setstate(state)
    return minhashes

class Race(tf.Module):
    def __init__(self, repetitions, concatenations, buckets, seed=314215):
        self.r = repetitions
        self.c = concatenations 
        self.b = buckets
        self.seed = seed
        self.arrays = tf.Variable(np.zeros(shape=(self.r, self.b)), dtype=tf.float64)
        return
    
    @tf.function
    def get_indices(self, x):
        hashes = minhash(x, self.r * self.c, self.b, self.seed)
        hashes = tf.reshape(hashes, (self.r, self.c))
        hashes = tf.strings.reduce_join(tf.as_string(hashes), axis=-1)
        # We don't salt the strings here because we are hashing only to get an integer
        # from the concatenation; we don't care about the randomness.
        indices = tf.cast(tf.strings.to_hash_bucket_fast(hashes, self.b), dtype=tf.int64)
        return tf.stack([tf.constant(np.arange(self.r), dtype=tf.int64), indices], axis=1)
    
    @tf.function
    def query(self, x):
        indices = self.get_indices(x)
        return tf.reduce_mean(tf.gather_nd(self.arrays, indices))
    
    @tf.function
    def index_then_query(self, x):
        indices = self.get_indices(x)
        update = tf.ones(shape=indices.shape[0], dtype=tf.float64)
        self.arrays.assign(tf.tensor_scatter_nd_add(self.arrays, indices, update))
        return tf.reduce_mean(tf.gather_nd(self.arrays, indices))
    
    @tf.function
    def query_then_index(self, x):
        indices = self.get_indices(x)
        score = tf.reduce_mean(tf.gather_nd(self.arrays, indices))
        update = tf.ones(shape=indices.shape[0], dtype=tf.float64)
        self.arrays.assign(tf.tensor_scatter_nd_add(self.arrays, indices, update))
        return score

########## TESTS ##########

# MINHASH TESTS

def test_minhash_right_num_hashes():
    """
    Check that it gives the right number of hashes (correct shape)
    """
    x = tf.constant([0,1,2,3,4,5,6], dtype=tf.int64)
    n_hashes = 10
    hash_range = 10000
    seed = 10
    hashes = minhash(x, n_hashes, hash_range, seed)
    assert(len(hashes.shape) == 1)
    assert(hashes.shape[0] == n_hashes)

def test_minhash_correlation_with_jaccard():
    """
    Check that when there are enough hashes, the number of hash collisions 
    between two tensors is similar to their jaccard similarity. 
    Show that higher similarity => more collisions.
    """
    x = tf.constant([0,1,2,3])
    y = tf.constant([0,1,2,4])
    z = tf.constant([0,1,5,6])
    xy_exp_col_rate = 0.6
    xz_exp_col_rate = 0.33
    err_margin = 0.05

    n_hashes = 1000
    hash_range = 10000
    seed = 10
    x_hashes = minhash(x, n_hashes, hash_range, seed)
    y_hashes = minhash(y, n_hashes, hash_range, seed)
    z_hashes = minhash(z, n_hashes, hash_range, seed)

    print(x_hashes)
    print(z_hashes)

    n_xy_collisions = np.count_nonzero(x_hashes == y_hashes)
    n_xz_collisions = np.count_nonzero(x_hashes == z_hashes)
    assert(n_xz_collisions < n_xy_collisions)
    assert(n_xy_collisions / n_hashes >= xy_exp_col_rate - err_margin)
    assert(n_xy_collisions / n_hashes <= xy_exp_col_rate + err_margin)
    assert(n_xz_collisions / n_hashes >= xz_exp_col_rate - err_margin)
    assert(n_xz_collisions / n_hashes <= xz_exp_col_rate + err_margin)

def test_minhash_seed():
    """
    Check that same seeds give same hashes, different seeds give different hashes.
    """
    x = tf.constant([0,1,2,3])
    n_hashes = 100
    hash_range = 10000
    seed = 10
    dif_seed = 11
    hashes = minhash(x, n_hashes, hash_range, seed)
    same_seed_hashes = minhash(x, n_hashes, hash_range, seed)
    dif_seed_hashes = minhash(x, n_hashes, hash_range, dif_seed)
    assert(tf.reduce_all(tf.equal(hashes, same_seed_hashes)) == True)
    assert(tf.reduce_all(tf.equal(hashes, dif_seed_hashes)) == False)

# RACE TESTS

def test_race_get_indices():
    """
    Check that get_indices gives the right number of indices, and each
    index is in the right range.
    """
    x = tf.constant([0,1,2,3])
    repetitions = 10
    concatenations = 3
    buckets = 1000
    race = Race(repetitions, concatenations, buckets)
    indices = race.get_indices(x)
    print(indices)
    assert(indices.shape == (repetitions, 2))
    assert(np.count_nonzero(indices[:,1] >= 0) == repetitions)
    assert(np.count_nonzero(indices[:,1] < buckets) == repetitions)
    assert(tf.reduce_all(tf.equal(indices[:,0], tf.constant(np.arange(repetitions)))) == True)

def test_race_score_trivial():
    """
    Check that after indexing the same tensor multiple times,
    a query into RACE with the same tensor will give a score
    that is equal to the number of times the tensor is indexed.
    """
    x = tf.constant([0,1,2,3])
    race = Race(repetitions=10, concatenations=3, buckets=1000)
    times_indexed = 100
    for _ in range(times_indexed):
        race.index_then_query(x)
    assert(race.query(x) == times_indexed)

def test_race_different_tensors():
    """
    Check that after indexing very different tensors multiple times,
    a query into RACE with the these tensors will give scores
    that are similar to the number of times each tensor is indexed,
    since we expect little to no hash collisions between them.
    """
    x = tf.constant([0,1,2,3])
    y = tf.constant([4,5,6,7])
    race = Race(repetitions=10, concatenations=1, buckets=1000)
    times_indexed = 100
    for _ in range(times_indexed):
        race.index_then_query(x)
        race.index_then_query(y)
    assert(race.query(x) < 1.2 * times_indexed)
    assert(race.query(x) >= times_indexed)
    assert(race.query(y) < 1.2 * times_indexed)
    assert(race.query(y) >= times_indexed)

def test_race_similar_tensors():
    """
    Check that after indexing very different tensors multiple times,
    a query into RACE with the these tensors will give a score
    that is between 1.2-1.8 times the number of times they are each
    indexed since we expect many hash collisions.
    """
    x = tf.constant([0,1,2,3])
    y = tf.constant([0,1,2,4])
    race = Race(repetitions=20, concatenations=1, buckets=1000)
    times_indexed = 100
    for _ in range(times_indexed):
        race.index_then_query(x)
        race.index_then_query(y)
    assert(race.query(x) < 1.8 * times_indexed)
    assert(race.query(x) > 1.2 * times_indexed)
    assert(race.query(y) < 1.8 * times_indexed)
    assert(race.query(y) > 1.2 * times_indexed)
