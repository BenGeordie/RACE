from typing import List
import tensorflow as tf
from race import Race
import pdb


class weighted_race():
    def __init__(self, race: Race, embedding_model: tf.Module, classifier_model: tf.Module, accept_first_n: int, score_threshold: float, accept_prob: float):

        self._n = tf.Variable(0, dtype=tf.int64)
        self.accept_first_n = accept_first_n 
        self.score_threshold = score_threshold
        self.accept_prob = accept_prob 
        self.embedding_model = embedding_model
        self.classifier_model = classifier_model
        self.race = race
#     def samples_seen(self):
#         """Returns the number of samples seen so far.
#         """
#         return self._n
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _sample_after_first_n(self, scores, y):
        # score_threshold <= 1.0
        # score_threshold - scores is in the range [-1 + score_threshold, score_threshold]
        # Thus, taking the ceiling results in 1 if score < threshold, 0 otherwise.
       # accepted_by_score_weights = tf.cast(tf.math.ceil(self.score_threshold - scores), dtype=tf.float32) 
        accepted_by_score_weights = tf.cast(tf.math.greater_equal(self.score_threshold, tf.cast(scores, tf.float32)), dtype=tf.float32)
        
        # Accepted by chance if random_num < accept_prob.
        # As above, taking the ceiling results in 1 if random_num < accept_prob, 0 otherwise.
        random_num = tf.random.uniform(shape=[tf.shape(scores)[0]])
       # accepted_by_chance = tf.math.ceil(tf.constant(self.accept_prob) - random_num)
        accepted_by_chance = tf.cast(tf.math.less_equal(random_num, tf.constant(self.accept_prob)), dtype=tf.float32)
        # If accepted, weight is 1 / accept_prob.
        accepted_by_chance_weights = tf.where(tf.cast(accepted_by_chance, dtype=tf.bool), tf.math.reciprocal_no_nan(self.accept_prob), [0.0])
        
        # If passing accepted_by_score_weights == 1.0, keep the weight. Otherwise, if accepted by random chance,
        # weight by 1 / accept_prob
        sampled_negative_weights = tf.where(tf.cast(accepted_by_score_weights, dtype=tf.bool), accepted_by_score_weights, accepted_by_chance_weights)
        # Accept if positive, weight accordingly otherwise.
       # tf.print('=====sample_after_first_n=====')
        
        return tf.where(tf.cast(y, dtype=tf.bool), [1.0], sampled_negative_weights)
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _weight_with_loss(self, x, y):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """

        pred = self.classifier_model(x)
        bce =  tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = bce(tf.reshape(y,(-1,1)),pred)
        loss = tf.reshape(loss,(-1,1))
        e = self.embedding_model(x)
        self._n.assign(self._n + tf.cast(tf.shape(x)[0], dtype=tf.int64))
        scores, n = self.race.score_weight(e, loss)
       # tf.print('=====Start _weight_with_loss=====')
       # tf.print('self._n inside weight_with_loss', self._n)
       # tf.print('n inside weight_with_loss', n)
       # tf.print('scores inside weight_with_loss', scores)
        #tf.print('=====Finish _weight_with_loss=====')
        return(scores)
        
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _weight_with_gradient(self, x, y):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """
        with tf.GradientTape() as tape:
            wghts = self.classifier_model.trainable_variables
      #      sh = [wghts[i].shape for i in range(len(wghts))]
   #         tf.print(sh)
            tape.watch(wghts)
            pred = self.classifier_model(x)
            # Loss
            bce =  tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            loss = bce(tf.reshape(y,(-1,1)),pred)
        # Compute Jacobian
        jacob = tape.jacobian(loss, wghts)
    #    tf.print('jacob len',len(jacob))
    #    tf.print('jacob -11',jacob[-11].shape)

        # compute gradient norm
        grad_norm = tf.zeros(tf.shape(x)[0])
        tf.print('len jacob',len(jacob))
        for i in range(len(jacob)):
           tf.print('loop num',i)
       #     if len(jacob[i].shape) == 3:
       #         grad_norm +=  tf.norm(jacob[i], ord=2, axis=(1,2))
       #     elif len(jacob[i].shape) == 2:
       #         grad_norm +=  tf.norm(jacob[i], ord=2, axis=(1))
           tf.print('jacob[i].shape',jacob[i].shape)
           tf.print('less?:', tf.less(len(jacob[i].shape),3))
          # tf.print('norm',tf.norm(jacob[34], ord=2, axis=(1,2)).shape)
           grad_norm = tf.cond(
                   tf.less(len(jacob[i].shape),3),
                   lambda: grad_norm + tf.norm(jacob[i], ord=2, axis=(1)),
           #        lambda: grad_norm+0.1,
                   lambda: grad_norm + tf.norm(jacob[i], ord=2, axis=(1,2))
                  # lambda: grad_norm+0.1,
                  # lambda: grad_norm
                  )
        grad_norm = tf.reshape(grad_norm,(-1,1))
        loss = tf.reshape(loss,(-1,1))

        e = self.embedding_model(x)
        self._n.assign(self._n + tf.cast(tf.shape(x)[0], dtype=tf.int64))
        scores, n = self.race.score_weight(e, grad_norm)

        return(scores)


    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def _no_weight(self, x, y):
        """
        Arguments:
            x: samples (both positive and negative)
            y: targets
        """
        e = self.embedding_model(x)
        self._n.assign(self._n + tf.cast(tf.shape(x)[0], dtype=tf.int64))
        scores, n = self.race.score_Noupdate(e)
       # tf.print('=====_no_weight=====')
       # tf.print('n inside _no_weight', self._n)
        return(scores)
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.int64)])
    def final_weight_loss(self, x, y):  
        
        cond = tf.less(self._n, tf.constant(self.accept_first_n, dtype=tf.int64))
        scores = tf.cond(
           # tf.less(self._n, tf.constant(self.accept_first_n, dtype=tf.int64)),
            cond,
            lambda: self._weight_with_loss(x, y),
            lambda: self._no_weight(x, y)
        )
       # tf.print('=====final_weight_loss=====')
       # tf.print('n inside final_weight_loss', self._n)
      #  tf.print('scores in final weight', scores)
       # tf.print('tf cond',cond)

        return tf.cond(
           # tf.less(self._n, tf.constant(self.accept_first_n, dtype=tf.int64)),
            cond,
            lambda: tf.ones(shape=tf.shape(x)[0]),
            lambda: self._sample_after_first_n(scores, y)
        )

    def final_weight_gradient(self, x, y):

        cond = tf.less(self._n, tf.constant(self.accept_first_n, dtype=tf.int64))
        scores = tf.cond(
           # tf.less(self._n, tf.constant(self.accept_first_n, dtype=tf.int64)),
            cond,
            lambda: self._weight_with_gradient(x, y),
            lambda: self._no_weight(x, y)
        )
       # tf.print('n inside final_weight', self._n)
       # tf.print('scores in final weight', scores)
       # tf.print('tf cond',cond)

        return tf.cond(
           # tf.less(self._n, tf.constant(self.accept_first_n, dtype=tf.int64)),
            cond,
            lambda: tf.ones(shape=tf.shape(x)[0]),
            lambda: self._sample_after_first_n(scores, y)
        )
#     return _weight
