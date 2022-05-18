import utils
import models
import tensorflow as tf

data = utils.load_movielens_csv("/Users/benitogeordie/Downloads/Movielenslatest_x1/data/train_contig.csv").batch(512)
embed_model = models.make_movielens_embedding_model()

data = utils.load_avazu_csv("/Users/benitogeordie/Downloads/Avazu_x4/data/train_contig_noid.csv").batch(512)
embed_model = models.make_avazu_embedding_model()

race = utils.make_default_race(embed_model)
weight_fn = utils.weight_with_race(race, embed_model, 10_000, 0.012, 0.1)

total_size = 0
for i, (x, y, w) in enumerate(utils.weight_and_filter(data, weight_fn)):
    print(i, end='\r', flush=True)
    if i >= 5000:
        break
    total_size += tf.shape(x)[0]
    # print(x)

print("downsampled size =", total_size)