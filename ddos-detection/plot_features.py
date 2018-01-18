import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_feature_from(filename, feature_name):
    normal = []
    botnets = []
    with open(filename, 'r+') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            data = line.strip().split(',')
            data = dict(zip(headers, data))
            if data['label'] == 'Botnet':
                botnets.append(float(data[feature_name]))
            elif data['label'] == 'Normal':
                normal.append(float(data[feature_name]))
    return np.array(normal), np.array(botnets)


def plot_histogram_of(filename, feature, save=False):
    name = filename.split('/')[1].split('.')[0]
    normal, botnets = get_feature_from(filename, feature)
    print('starting', name, feature)

    n_normal, bins_normal, patches_normal = plt.hist(
        normal, 50, facecolor='blue', alpha=.5)
    n_botnet, bins_botnet, patches_botnet = plt.hist(
        botnets, 50, facecolor='red', alpha=.5)
    plt.title('{} on {}'.format(name, feature))
    if save:
        plt.savefig('figures/{}-{}.png'.format(name, feature))
    else:
        plt.show()


if __name__ == '__main__':
    filename = 'aggregated_binetflows/capture20110818-2.aggregated.csv'
    files = [
        'aggregated_binetflows/capture20110818-2.aggregated.csv',
        'aggregated_binetflows/capture20110818.aggregated.csv',
        'aggregated_binetflows/capture20110815.aggregated.csv'
    ]
    features = [
        'std_packet',
        'std_bytes',
        'std_time',
        'std_dstport',
        'std_bytes',
    ]
    for f in files:
        for feature in features:
            plot_histogram_of(f, feature, save=True)
