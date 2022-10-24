import tensorflow as tf
import numpy as np
from lsh_functions import PStableHash
import pdb

# Tested lightly
class Race(tf.Module):
    def __init__(self, repetitions: int, concatenations: int, buckets: int, hash_module: tf.Module, total_samples: int):
        """Race Constructor.
        Arguments:
            repetitions: integer. Number of rows in RACE.
            concatenations: integer. Number of hashes concatenated to get column index for each row.
            buckets: integer. Number of columns in RACE.
            hash_module: LSH module from lsh_functions.py.
        """

        if not hasattr(hash_module, "hash") or not hasattr(hash_module, "_num_hashes"):
            raise ValueError("Hash module must have a hash() method and a '_num_hashes' attribute.")
        
        if hash_module._num_hashes != repetitions * concatenations:
            raise ValueError(f"hash_module._num_hashes ({hash_module._num_hashes}) must equal " \
                             + f"repetitions * concatenations ({repetitions * concatenations})")

        self._r = repetitions
        self._c = concatenations 
        self._b = buckets
        self._hash_module = hash_module
        self._arrays = tf.Variable(np.zeros(shape=(self._r, self._b)), dtype=tf.float64)
        self._total_samples = tf.constant(total_samples, dtype=tf.int64)
        self._n = tf.Variable(0, dtype=tf.int64)
        self._initial_n = tf.Variable(0, dtype=tf.int64)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _concatenate_last_axis(self, x: tf.Tensor):
        """Helper function to "concatenate" values along the last axis of a tensor.
        The values are concatenated as a string then rehashed into a number in range
        [0, buckets).

        Arguments:
            x: tensor of hash values with shape (n_samples, concatenations). 
        Returns:
            A tensor of hash values with shape (n_samples,)
        """
        conc = tf.strings.reduce_join(tf.as_string(x), axis=-1)
        conc = tf.strings.to_hash_bucket_fast(conc, self._b)
        return tf.cast(conc, dtype=tf.int64)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def _get_indices(self, x):
        """Helper function to get the row and column indices that a tensor is hashed to.
        Arguments:
            x: Tensor of samples to be scored and indexed, with shape (n_samples, sample_dim).
        Returns:
            A tensor of indices that each sample is hashed to, with shape (n_samples, repetitions, 2).
                The last dimension is 2 because there are both row and column indices.
        """
        hashes = self._hash_module.hash(x) 
        
        # Split hashes into _r groups (1 group per row), then concatenate the hashes in each group.
        hash_groups = tf.split(hashes, self._r, axis=-1)
        col_indices = tf.stack([self._concatenate_last_axis(h) for h in hash_groups], axis=-1)
        
        # For each column index, we also need the corresponding row index.
        # The row indices are simply the range [0, _r)
        row_indices = np.arange(self._r)

        # map_fn maps across axis 0. This assumes that dim-0 is the number of samples. 
        # This is a safe assumption because our embedding model will always output a 2D array,
        # even if the samples are not batched.
        return tf.map_fn(lambda idxs: tf.stack([row_indices, idxs], axis=-1), col_indices)
    
   
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def score(self, x):
        """Computes the RACE score of received samples and updates the RACE data structure.
        Arguments: 
            Tensor of samples to be scored and indexed, with shape (n_samples, sample_dim).
        Returns:
            Tensor of RACE scores for each sample, with shape (n_samples,).
        """
        indices = self._get_indices(x)
        
        # This assumes that dim-0 is the number of samples.
        # This is a safe assumption because our embedding model will always output a 2D array,
        # even if the samples are not batched.
        self._n.assign(self._n + tf.cast(tf.shape(x)[0], dtype=tf.int64))
        
        # For each sample, increment the counters at the locations that they are hashed to.
        # Shape is n_samples x repetitions.
        update = tf.ones(shape=tf.shape(indices)[:-1], dtype=tf.float64)
        self._arrays.assign(tf.tensor_scatter_nd_add(self._arrays, indices, update))

        # The score for each sample is the average count across every row
        # divided by the number of elements that had been previously indexed.
        score = tf.reduce_mean(tf.gather_nd(self._arrays, indices), axis=-1)
        # Prevent division by 0.
        one_over_n = tf.math.reciprocal_no_nan(tf.cast(self._n, dtype=tf.float64))
      #  tf.print('score before',score)
    #    tf.print('n inside race',self._n)
        score *= one_over_n
    #    tf.print('score after',score)
    #    tf.print('score shape',score.shape)

        return score, self._n

    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)])
    def score_weight(self, x, weight):
        """Computes the RACE score of received samples and updates the RACE data structure.
        Arguments: 
            Tensor of samples to be scored and indexed, with shape (n_samples, sample_dim).
        Returns:
            Tensor of RACE scores for each sample, with shape (n_samples,).
        """
       # tf.print('====Start score_weight in race.py====')
        indices = self._get_indices(x)
       # tf.print('indices',indices)
       # tf.print('indiecs shape',indices.shape)
        # This assumes that dim-0 is the number of samples.
        # This is a safe assumption because our embedding model will always output a 2D array,
        # even if the samples are not batched.
        self._n.assign(self._n + tf.cast(tf.shape(x)[0], dtype=tf.int64))
        
        # For each sample, increment the counters at the locations that they are hashed to.
        # Shape is n_samples x repetitions.
        #update = tf.ones(shape=tf.shape(indices)[:-1], dtype=tf.float64)
        inv_weight = tf.math.reciprocal_no_nan(weight)
       # tf.print('weight',weight)
       # tf.print('inv_weight',inv_weight)
       # tf.print('inv weight shape',inv_weight.shape)
       # tf.print('self.r',self._r)
        update = tf.cast(tf.tile(inv_weight,tf.constant([1,self._r])), dtype=tf.float64)
       # tf.print('update', update)
       # tf.print('update shape', update.shape)
       # tf.print('before update array',self._arrays)
       # tf.print('array shape',self._arrays.shape)
        self._arrays.assign(tf.tensor_scatter_nd_add(self._arrays, indices, update))
       # tf.print('updated array',self._arrays)
        # The score for each sample is the average count across every row
        # divided by the number of elements that had been previously indexed.
        score = tf.reduce_mean(tf.gather_nd(self._arrays, indices), axis=-1)
        # Prevent division by 0.
        one_over_n = tf.math.reciprocal_no_nan(tf.cast(self._n, dtype=tf.float64))
       # tf.print('score_weight before',score)
      #  tf.print('n inside race',self._n)
        score *= one_over_n
       # tf.print('score_weight after',score)
       # tf.print('score shape',score.shape)
        return score, self._n


    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def score_Noupdate(self, x):
        """Computes the RACE score of received samples and updates the RACE data structure.
        Arguments:
            Tensor of samples to be scored and indexed, with shape (n_samples, sample_dim).
        Returns:
            Tensor of RACE scores for each sample, with shape (n_samples,).
        """
        indices = self._get_indices(x)

        # This assumes that dim-0 is the number of samples.
        # This is a safe assumption because our embedding model will always output a 2D array,
        # even if the samples are not batched.
        self._n.assign(self._n + tf.cast(tf.shape(x)[0], dtype=tf.int64))

        # For each sample, increment the counters at the locations that they are hashed to.
        # Shape is n_samples x repetitions.
       # update = tf.ones(shape=tf.shape(indices)[:-1], dtype=tf.float64)
       # self._arrays.assign(tf.tensor_scatter_nd_add(self._arrays, indices, update))

        # The score for each sample is the average count across every row
        # divided by the number of elements that had been previously indexed.
        score = tf.reduce_mean(tf.gather_nd(self._arrays, indices), axis=-1)
        # Prevent division by 0.
       # one_over_n = tf.math.reciprocal_no_nan(tf.cast(self._n, dtype=tf.float64))
      #  tf.print('score before',score)
    #    tf.print('n inside race',self._n)
       # score *= one_over_n # ???
    #    tf.print('score after',score)
    #    tf.print('score shape',score.shape)

        return score, self._n        

    def samples_seen(self):
        """Returns the number of samples seen so far.
        """
        return self._n
    
    def summary(self):
        """Prints some statistics about the RACE data structure.
        """
        mean_nonzeros_per_row = tf.math.count_nonzero(self._arrays) / self._r
        
        total_nonzeros = tf.math.count_nonzero(self._arrays)
        nonzero_counter_mean = tf.truediv(tf.reduce_sum(self._arrays), tf.cast(total_nonzeros, dtype=tf.float64))
        
        nonzero_indices = tf.where(self._arrays)
        nonzero_buckets = tf.gather_nd(self._arrays, nonzero_indices)
        
        nonzero_counter_stdev = tf.math.reduce_std(nonzero_buckets)
        nonzero_counter_min = tf.reduce_min(nonzero_buckets)
        nonzero_counter_max = tf.reduce_max(nonzero_buckets)

        return {
            "mean_nonzeros_per_row": mean_nonzeros_per_row.numpy(),
            "nonzero_counter_mean": nonzero_counter_mean.numpy(),
            "nonzero_counter_stdev": nonzero_counter_stdev.numpy(),
            "nonzero_counter_min": nonzero_counter_min.numpy(),
            "nonzero_counter_max": nonzero_counter_max.numpy()
        }
    
