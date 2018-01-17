import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_feature_from(filename, feature_name, from_botnet=False):
    x = []
    with open(filename, 'r+') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            data = line.strip().split(',')
            data = dict(zip(headers, data))
            if from_botnet and data['label'] == 'Botnet':
                x.append(float(data[feature_name]))
            elif not from_botnet and data['label'] == 'Normal':
                x.append(float(data[feature_name]))
    return np.array(x)


def plot_histogram_of(filename, feature):
    n_normal, bins_normal, patches_normal = plt.hist(
            get_feature_from(filename, feature, from_botnet=False),
                facecolor='blue', alpha=.5)
    n_botnet, bins_botnet, patches_botnet = plt.hist(
            get_feature_from(filename, feature, from_botnet=True),
                facecolor='red', alpha=.5)
    plt.savefig('img.png')


if __name__ == '__main__':
    filename = 'aggregated_binetflows/capture20110815.aggregated.csv'
    plot_histogram_of(filename, 'std_bytes')

