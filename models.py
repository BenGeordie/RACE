from typing import List
import tensorflow as tf
import tensorflow.keras as keras

# Tested lightly

def make_criteo_nn(embedding_model: tf.Module, hidden_layer_dims: List[int]):
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

    optimizer = keras.optimizers.Adam()
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

    if isinstance(layer, tf.keras.layers.Embedding):
        layer_inputs = tf.math.mod(layer_inputs, layer.input_dim)
        output_dim = layer.output_dim

    else: # The layer is a one-hot encoding layer
        layer_inputs = tf.math.mod(layer_inputs, layer.num_tokens)
        output_dim = layer.num_tokens

    # return layer(layer_inputs)
    final_shape = (-1, output_dim)
    return tf.reshape(layer(layer_inputs), shape=final_shape)

def make_criteo_embedding_model():
    """
    Creates a TensorFlow model that takes in samples from the Criteo clickthrough dataset
    and embeds the categorical entries.

    We use embedding layers for entries with > 100 unique values, one-hot encoding layers otherwise.
    For entries with >1000 entries, we have sqrt(unique values) embedding table entries.
    """
    embedding_layers = [
        keras.layers.Embedding(input_dim=100, output_dim=70, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=531, output_dim=50, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=600, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=400, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=267, output_dim=50, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=15, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=110, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=563, output_dim=50, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=3, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=200, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=200, output_dim=70, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=600, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=150, output_dim=70, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=26, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=300, output_dim=70, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=500, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=10, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=250, output_dim=70, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.Embedding(input_dim=100, output_dim=70, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=3, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=600, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=15, output_mode='one_hot'),
        keras.layers.CategoryEncoding(num_tokens=15, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=200, output_dim=100, 
                                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.)),
        keras.layers.CategoryEncoding(num_tokens=69, output_mode='one_hot'),
        keras.layers.Embedding(input_dim=200, output_dim=100, 
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