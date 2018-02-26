import matplotlib
import itertools
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def convert_to_greek(line):
    line.replace('std', '$\sigma$')
    line.replace('entropy', '$\mathcal{S}$')
    line.replace('n_', '$\mathcal{N}$_')
    line.replace('avg', '$\mathcal{\bar{W}}$')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def best_features(forest, feature, name, feature_names):
    plt.figure(figsize=(10, 5))
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    plt.title('Importances for {}'.format(name))
    plt.bar(range(feature.shape[1]), importances[indices],
            color='r', yerr=std[indices], align='center')
    features = [convert_to_greek(feature_names[i]) for i in indices]
    plt.xticks(range(feature.shape[1]), features, rotation=90)
    plt.xlim([-1, feature.shape[1]])
    plt.show()
    return indices


def plot_f1_per_interval(f1_scores, name, intervals, save=False):
    plt.plot(intervals, f1_scores)
    plt.xticks(intervals)
    plt.xlabel('Interval (seconds)')
    plt.ylabel('f1_score')
    plt.title(name)
    if save:
        plt.save(name + '.png')
    else:
        plt.show()


def plot_rf_estimators(f1_scores, name, save=False):
    estimator_counts = [10, 50, 100, 200, 300, 500, 700]
    plt.plot(estimator_counts, f1_scores)
    plt.xticks(estimator_counts)
    plt.xlabel('Estimator counts')
    plt.ylabel('f1_score')
    plt.title(name)
    if save:
        plt.save(name + '.png')
    else:
        plt.show()


def plot_multilabel_roc(precision, recall, name, save=False):
    plt.plot(recall['micro'],
             precision['micro'],
             color='gold',
             lw=2,
             label='Micro-Average')
    colors = ['black', 'red', 'blue', 'green']
    classes = ['Normal', 'DDOS', 'SPAM', 'IRC']
    for i, color in zip(range(4), colors):
        # TODO: Actually tell which class.
        plt.plot(recall[i], precision[i], color=color, label='Class ' + classes[i])
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.legend()
    plt.show()


def plot_roc_curve(fpr, tpr, auc, name, label, color, save=False):
    plt.plot(fpr, tpr, label=label, color=color)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if save:
        plt.save(name + '.png')
    else:
        pass


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
    feature = convert_to_greek(feature)
    name = filename.split('/')[1].split('.')[0]
    normal, botnets = get_feature_from(filename, feature)
    n_normal, bins_normal, patches_normal = plt.hist(
        normal, 50, facecolor='blue', alpha=.5)
    n_botnet, bins_botnet, patches_botnet = plt.hist(
        botnets, 50, facecolor='red', alpha=.5)
    plt.title('{} on {}'.format(name, feature))
    if save:
        plt.savefig('figures/{}-{}.png'.format(name, feature))
    # else:
    #     plt.show()


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
