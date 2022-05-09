from typing import List
import tensorflow as tf
import tensorflow.keras as keras
import pdb

# Tested lightly

def make_criteo_nn(embedding_model: tf.Module, hidden_layer_dims: List[int], lr: tf.float32):
    """Creates a neural network with a trainable embedding model as its base with hidden layers on top of it.
    The network trains to minimize BinaryCrossentropy loss and is evaluated with the AUC metric
    Arguments:
        embedding_model: Trainable embedding model to be used as the network's base.
        hidden_layer_dims: List of integers. A sequence of hidden layer dimensions to be stacked on
            top of the embedding model base.
    """
    hidden_layers = [keras.layers.Dense(dim, activation=tf.nn.relu) for dim in hidden_layer_dims]
    
    final_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    layers = keras.Sequential([
        *hidden_layers,
        final_layer
    ])

    inputs = keras.Input(shape=(39,), dtype=tf.float32)
    x = embedding_model(inputs)
    outputs = layers(x)
    model = keras.Model(inputs, outputs)

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

    We use embedding layers for entries with > 100 unique values, one-hot encoding layers otherwise.
    For entries with >1000 entries, we have sqrt(unique values) embedding table entries.

    Categorical feature levels for train data:
    0 : 1460 | 1 : 558 | 2 : 413422 | 3 : 248541 | 4 : 305 | 5 : 21 | 6 : 12190 | 7 : 633 | 8 : 3 | 9 : 54710| 10 : 5348 | 11 : 409747 | 12 : 3180 | 13 : 27 | 14 : 12498 | 15 : 365809 | 16 : 10 | 17 : 4932 | 18 : 2094 | 19 : 4 |  20 : 397979 | 21 : 18 | 22 : 15 | 23 : 88606 | 24 : 96 | 25 : 64071  
    input_dim = cat_level + 1 add one to consider OOV
    output_dim = 6*(input_dim)**.025
    """
    embedding_layers = [
        keras.layers.Embedding(input_dim=1461, output_dim=38,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=559, output_dim=30, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=413423, output_dim=153, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=248542, output_dim=134, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=306, output_dim=26, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=22, output_dim=13,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=12191, output_dim=64, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=634, output_dim=31,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=4, output_dim=8,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=54711, output_dim=92, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=5349, output_dim=52, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=409748, output_dim=152, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=3181, output_dim=46, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=28, output_dim=14, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=12499, output_dim=64,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=365900, output_dim=148, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=11, output_dim=11, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=4933, output_dim=51,   
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=2095, output_dim=41,  
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=5, output_dim=9,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=397980, output_dim=151, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=19, output_dim=13,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=16, output_dim=12,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=88607, output_dim=104, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=97, output_dim=19,
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=64072, output_dim=96, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
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
