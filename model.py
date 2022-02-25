import tensorflow as tf
import tensorflow.keras as keras


def concat_helper(layer, inputs, index):
    layer_inputs = tf.gather(inputs, [index], axis=1)
    if isinstance(layer, tf.keras.layers.Embedding):
        layer_inputs = tf.math.mod(layer_inputs, layer.input_dim)
        output_dim = layer.output_dim
    else:
        layer_inputs = tf.math.mod(layer_inputs, layer.num_tokens)
        output_dim = layer.num_tokens
    final_shape = (-1, output_dim)
    return tf.reshape(layer(layer_inputs), final_shape)


def make_criteo_model():
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

    hidden_layer_1 = keras.layers.Dense(500, activation=tf.nn.relu)
    hidden_layer_2 = keras.layers.Dense(500, activation=tf.nn.relu)
    hidden_layer_3 = keras.layers.Dense(500, activation=tf.nn.relu)
    final_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    layers = keras.Sequential([
        hidden_layer_1,
        hidden_layer_2,
        hidden_layer_3,
        final_layer
    ])

    inputs = keras.Input(shape=(39,), dtype=tf.float32)
    x = keras.layers.Concatenate()([
        tf.gather(inputs, range(0,13), axis=1),
        *[concat_helper(layer, inputs, i + 13) for i, layer in enumerate(embedding_layers)]
    ])
    outputs = layers(x)
    model = keras.Model(inputs, outputs)

    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()
    metric = keras.metrics.BinaryAccuracy()

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[metric])
    return model

