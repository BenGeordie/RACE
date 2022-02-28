import utils
from models import make_criteo_nn, make_criteo_embedding_model
from lsh_functions import PStableHash
from race import Race

# Untested
def sample_pipeline():
    
    # Load data and split into train, validation, and test datasets.
    data = utils.load_csv("path/to/criteo.csv")
    train_ds, val_ds, test_ds = utils.split_train_val_test(data)

    # RACE hyper parameters
    repetitions = 100
    concatenations = 2
    buckets = 1_000_000
    p = 1.0
    seed = 314152
    
    embedding_model = make_criteo_embedding_model()
    hash_module = PStableHash(embedding_model.output_shape[1], num_hashes=repetitions * concatenations, p=p, seed=seed)
    race = Race(repetitions, concatenations, buckets, hash_module)

    # Weighting function hyper parameters
    accept_first_n = 1_000_000
    score_threshold = 0.05
    accept_prob = 0.05
    
    weight_fn = utils.weight_with_race(race, embedding_model, accept_first_n, score_threshold, accept_prob)
    filtered_weighted_train_ds = utils.weight_and_filter(train_ds, weight_fn)

    # Dimensions of neural network hidden layers.
    hidden_layer_dims = [10_000, 10_000, 10_000]
    nn = make_criteo_nn(embedding_model, hidden_layer_dims)

    # Run!
    nn.fit(filtered_weighted_train_ds)
    nn.evaluate(test_ds)
    nn.evaluate(val_ds)


