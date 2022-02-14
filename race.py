import tensorflow as tf
import random
import sys

@tf.function
def minhash(x: tf.Tensor, r_repetitions: int, c_concatenations: int, b_buckets: int, seed: int=314152):
    state = random.getstate()
    random.seed(seed)
    n_hashes = r_repetitions * c_concatenations
    hashes = [
        x * random.randint(0, sys.maxsize) + random.randint(0, sys.maxsize) 
        for _ in range(n_hashes)]
    minhashes = tf.stack([tf.reduce_min(t, axis=1) for t in hashes])
    minhashes = tf.strings.reduce_join(tf.as_string(minhashes), axis=0)
    # We don't salt the strings we are only using the hash function 
    # to get a number, not for randomness.
    minhashes = tf.strings.to_hash_bucket_fast(minhashes, b_buckets) 
    random.setstate(state)
    return minhashes

class Race(tf.Module):
    def __init__(self):
        return