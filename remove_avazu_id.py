def remove_avazu_id(path):
    with open(path, 'r') as i:
        with open(path[:-4] + '_noid.csv', 'w') as o:
            for line in i:
                o.write(','.join(line.split(',')[1:]))

remove_avazu_id('/Users/benitogeordie/Downloads/Avazu_x4/train_contig.csv')
remove_avazu_id('/Users/benitogeordie/Downloads/Avazu_x4/test_contig.csv')
remove_avazu_id('/Users/benitogeordie/Downloads/Avazu_x4/valid_contig.csv')