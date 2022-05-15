from make_contiguous_catfeatures_base import make_contiguous

base_path = '/Users/benitogeordie/Downloads/Movielenslatest_x1/'

make_contiguous(
    small_csv=base_path + 'train_5.csv',
    train_csv=base_path + 'train.csv',
    test_csv=base_path + 'test.csv',
    valid_csv=base_path + 'valid.csv', 
    start_idx=1,
    out_prefix=base_path,
    use_gpu=False,
)