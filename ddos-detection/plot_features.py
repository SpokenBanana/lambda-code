import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_f1_per_interval(f1_scores, name, save=False):
    intervals = [10, 20, 30, 60, 120, 180]
    plt.plot(intervals, f1_scores)
    plt.xticks(intervals)
    plt.xlabel('Interval (seconds)')
    plt.ylabel('f1_score')
    plt.title(name)
    if save:
        plt.save(name + '.png')
    else:
        plt.show()


def plot_roc_curve(fpr, tpr, auc, name, save=False):
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC={:.2} for {}'.format(auc, name))
    if save:
        plt.save(name + '.png')
    else:
        plt.show()


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
    directory = 'minute_aggregated'
    files = [
        'capture20110818-2.aggregated.csv',
        'capture20110818.aggregated.csv',
        'capture20110815.aggregated.csv'
    ]
    files = ['ddos.featureset.csv']
    files = ['{}/{}'.format(directory, f) for f in files]
    features = [
        'std_packet',
        'std_bytes',
        'std_time',
        'entropy_dstport',
        'entropy_srcport',
        'entropy_srcip',
        'entropy_dstip',
        'entropy_sports>1024',
        'entropy_sports<1024',
        'entropy_dports>1024',
        'entropy_dports<1024',
        'entropy_state',
        'std_bytes',
        # TODO: Add the rest of the 17 features here.
    ]
    for f in files:
        for feature in features:
            plot_histogram_of(f, feature, save=True)
