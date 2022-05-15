from typing import List
import tensorflow as tf
import tensorflow.keras as keras
import pdb

# Tested lightly

def make_clickthrough_nn(embedding_model: tf.Module, hidden_layer_dims: List[int], lr: tf.float32):
    """Creates a neural network with a trainable embedding model as its base with hidden layers on top of it.
    The network trains to minimize BinaryCrossentropy loss and is evaluated with the AUC metric
    Arguments:
        embedding_model: Trainable embedding model to be used as the network's base.
        hidden_layer_dims: List of integers. A sequence of hidden layer dimensions to be stacked on
            top of the embedding model base.
    """
    hidden_layers = [keras.layers.Dense(dim, activation=tf.nn.relu) for dim in hidden_layer_dims]
    
    final_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    model = keras.Sequential([
        embedding_model,
        *hidden_layers,
        final_layer
    ])

    # inputs = keras.Input(shape=(39,), dtype=tf.float32)
    # x = embedding_model(inputs)
    # outputs = layers(x)
    # model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    metric_auc = keras.metrics.AUC()
    metric_logloss = keras.metrics.BinaryCrossentropy()
    metric_acc = keras.metrics.Accuracy()
    loss_fn = keras.losses.BinaryCrossentropy()

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[metric_auc, metric_logloss, metric_acc])
    return model

from tensorflow import keras

# Tested lightly

def _embed(layer: tf.Module, inputs: tf.Tensor, index: int):
    """
    Helper function that embeds "index"-th entry of each sample in "inputs"
    using embedding layer "layer".
    """
    layer_inputs = tf.gather(inputs, [index], axis=-1)

    #if isinstance(layer, tf.keras.layers.Embedding):
   #     layer_inputs = tf.math.mod(layer_inputs, layer.input_dim)
   #     output_dim = layer.output_dim

   # else: # The layer is a one-hot encoding layer
   #     layer_inputs = tf.math.mod(layer_inputs, layer.num_tokens)
   #     output_dim = layer.num_tokens
    
    output_dim = layer.output_dim
    # return layer(layer_inputs)
    final_shape = (-1, output_dim)
    return tf.reshape(layer(layer_inputs), shape=final_shape)

def make_criteo_embedding_model():
    """
    Creates a TensorFlow model that takes in samples from the Criteo clickthrough dataset
    and embeds the categorical entries.

    Number of unique values for each categorical feature in train data:
    0 : 1460 | 1 : 558 | 2 : 413422 | 3 : 248541 | 4 : 305 | 5 : 21 | 6 : 12190 | 7 : 633 | 8 : 3 | 9 : 54710| 10 : 5348 | 11 : 409747 | 12 : 3180 | 13 : 27 | 14 : 12498 | 15 : 365809 | 16 : 10 | 17 : 4932 | 18 : 2094 | 19 : 4 |  20 : 397979 | 21 : 18 | 22 : 15 | 23 : 88606 | 24 : 96 | 25 : 64071  
    input_dim = cat_level + 1 add one to consider OOV
    output_dim = 6*(input_dim)**.025
    """
    n_uniques = [
        1460, # 0 
        558, # 1
        413422, # 2
        248541, # 3
        305, # 4
        21, # 5
        12190, # 6
        633, # 7
        3, # 8
        54710, # 9
        5348, # 10
        409747, # 11
        3180, # 12
        27, # 13
        12498, # 14
        365809, # 15
        10, # 16
        4932, # 17
        2094, # 18
        4, # 19
        397979, # 20 
        18, # 21
        15, # 22
        88606, # 23
        96, # 24
        64071, #25
    ]

    embedding_layers = [
        keras.layers.Embedding(input_dim=n_unique + 1, output_dim=6 * (n_unique ** 0.25),   
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.))
        for n_unique in n_uniques
    ]

    inputs = keras.Input(shape=(39,), dtype=tf.float32)
    int_inputs = tf.reshape(tf.gather(inputs, range(0,13), axis=-1), (-1, 13))
    outputs = keras.layers.Concatenate()([
        # First 13 entries in a Criteo sample are integral values.
        int_inputs,
        # The next 26 are categorical values that need to be embedded.
        *[_embed(layer, inputs, i + 13) for i, layer in enumerate(embedding_layers)]
    ])

    return keras.Model(inputs, outputs)

def make_avazu_embedding_model():
    """
    Creates a TensorFlow model that takes in samples from the Avazu clickthrough dataset
    and embeds the categorical entries.

    Categorical feature levels for train data:
    # TODO FIX THIS
    0 : 1460 | 1 : 558 | 2 : 413422 | 3 : 248541 | 4 : 305 | 5 : 21 | 6 : 12190 | 7 : 633 | 8 : 3 | 9 : 54710| 10 : 5348 | 11 : 409747 | 12 : 3180 | 13 : 27 | 14 : 12498 | 15 : 365809 | 16 : 10 | 17 : 4932 | 18 : 2094 | 19 : 4 |  20 : 397979 | 21 : 18 | 22 : 15 | 23 : 88606 | 24 : 96 | 25 : 64071  
    input_dim = cat_level + 1 add one to consider OOV
    output_dim = 40 based on results from https://arxiv.org/pdf/1904.04447.pdf
    """
    n_uniques = [
        1460, # 0 
        558, # 1
        413422, # 2
        248541, # 3
        305, # 4
        21, # 5
        12190, # 6
        633, # 7
        3, # 8
        54710, # 9
        5348, # 10
        409747, # 11
        3180, # 12
        27, # 13
        12498, # 14
        365809, # 15
        10, # 16
        4932, # 17
        2094, # 18
        4, # 19
        397979, # 20 
        18, # 21
        15, # 22
        88606, # 23
        96, # 24
        64071, #25
    ]

    embedding_layers = [
        keras.layers.Embedding(input_dim=n_unique + 1, output_dim=6 * (n_unique ** 0.25),   
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.))
        for n_unique in n_uniques
    ]

    inputs = keras.Input(shape=(39,), dtype=tf.float32)
    int_inputs = tf.reshape(tf.gather(inputs, range(0,13), axis=-1), (-1, 13))
    outputs = keras.layers.Concatenate()([
        # First 13 entries in a Criteo sample are integral values.
        int_inputs,
        # The next 26 are categorical values that need to be embedded.
        *[_embed(layer, inputs, i + 13) for i, layer in enumerate(embedding_layers)]
    ])

    return keras.Model(inputs, outputs)
