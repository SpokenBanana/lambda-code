"""Use machine learning on the new files.

Use for ROC curves:
    proba = clf.predict_proba(test_features).
    precision, recall, pr_threshold = precision_recall_curve(test_labels,
                                                             proba[:, 1])
    fpr, tpr, _ = roc_curve(test_labels, proba)  # proba[:, 1] for sklearn
    # roc_auc goes in the title
    roc_auc = auc(fpr, tpr)

    # Plot fpr vs tpr
"""
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.utils import plot_model
from sklearn import tree
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, \
    f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix, \
    average_precision_score
import os
from absl import app
from absl import flags
from utils import best_features
import numpy as np
from summarizer import Summarizer

# TODO: Remove it.
from plot_features import plot_multilabel_roc

# TODO: Make an easier way to generate data into latex tables.
LATEX_TABLE = """
\\begin{table}
\\end{table}
"""


def perform_search(params, feature, labels):
    clf = GridSearchCV(RandomForestClassifier(), params)

    # Train it.
    clf.fit(feature, labels)

    print(clf.best_params_)


def get_files(directory):
    files = os.listdir(directory)
    return ['{}/{}'.format(directory, name) for name in files]


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def standardize(x, mean, std):
    return (x - mean) / std


def get_roc_metrics(clf, features, labels, sklearn=True):
    proba = clf.predict_proba(features)
    if sklearn:
        precision, recall, pr_threshold = precision_recall_curve(labels,
                                                                 proba[:, 1])
    else:
        precision, recall, pr_threshold = precision_recall_curve(labels,
                                                                 proba)
    if sklearn:
        fpr, tpr, _ = roc_curve(labels, proba[:, 1])  # proba[:, 1] for sklearn
    else:
        fpr, tpr, _ = roc_curve(labels, proba)

    # roc_auc goes in the title
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# BOOM single line
def sample_feature_label(feature, label, n):
    return zip(*random.sample(list(zip(feature, label)), n))


def get_feature_labels(filename):
    features = []
    labels = []
    with open(filename, 'r+') as f:
        f.readline()
        for line in f:
            info = line.strip().split(',')
            features.append([float(x) for x in info[:-1]])
            labels.append(1 if info[-1] == 'Botnet' else 0)
    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=.3, random_state=42)
    return xtrain, xtest, ytrain, ytest


def get_ahead_feature_labels(filename, feature_names, steps_ahead=1):
    features = []
    labels = []

    queue = []

    with open(filename, 'r+') as f:
        header = f.readline().strip().split(',')

        for i in range(steps_ahead):
            prev_info = f.readline().strip().split(',')
            prev_info = dict(zip(header, prev_info))
            queue.append(prev_info)

        for line in f:
            info = line.strip().split(',')
            if len(info) < 2:
                # This is a separator, meaning we are at the start of a new
                # file and so the previous information is useless.
                queue = []
                for i in range(steps_ahead):
                    prev_info = f.readline().strip().split(',')
                    prev_info = dict(zip(header, prev_info))
                    queue.append(prev_info)
                continue

            info = dict(zip(header, info))
            prev_info = queue.pop(0)

            features.append([float(prev_info[name]) for name in feature_names])

            current = 1 if info['label'] == 'Botnet' else 0
            prev = 1 if prev_info['label'] == 'Botnet' else 0
            queue.append(info)

            if current == prev and current == 0:
                # labels.append([1, 0, 0, 0])
                labels.append(0)
            elif current == prev and current == 1:
                # labels.append([0, 1, 0, 0])
                labels.append(1)
            elif current != prev and current == 0:
                # labels.append([0, 0, 1, 0])
                labels.append(2)
            elif current != prev and current == 1:
                # labels.append([0, 0, 0, 1])
                labels.append(3)
    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=.3, random_state=42)
    return xtrain, xtest, ytrain, ytest


def get_specific_features_from(filename, feature_names=None, use_bots=False,
                               use_attack=False, has_bots_and_attack=False,
                               sample=False, norm_and_standardize=False):
    features = []
    labels = []
    with open(filename, 'r+') as f:
        header = f.readline().strip().split(',')
        for line in f:
            info = line.strip().split(',')
            data = dict(zip(header, info))
            if feature_names:
                features.append(
                    [float(data[feature]) for feature in feature_names])
            else:
                features.append([float(x) for x in info[:-1]])
            if use_bots:
                # For bots
                bots = ['Normal', 'Neris', 'Rbot', 'Virut', 'Menti', 'Sogou',
                        'Murlo', 'NSIS.ay']
                labels.append(bots.index(info[-2]))
            if use_attack:
                attacks = ['Normal', 'ddos', 'spam', 'irc']
                entry = [1, 0, 0, 0]
                if info[-3] == 'Botnet':
                    for attack in info[-1].split('+'):
                        entry[attacks.index(attack)] = 1
                    entry[0] = 0
                labels.append(entry)
            if has_bots_and_attack or 'all' in filename:
                labels.append(1 if info[-3] == 'Botnet' else 0)
            elif not use_bots and not use_attack:
                labels.append(1 if info[-1] == 'Botnet' else 0)
    if sample:
        botnet_feat, botnet_label = zip(*[
            x for x in zip(features, labels) if x[1] == 1])
        normal_feat, normal_label = zip(*[
            x for x in zip(features, labels) if x[1] != 1])
        normal_feat, normal_label = sample_feature_label(
            normal_feat, normal_label, len(botnet_feat))
        features = normal_feat + botnet_feat
        labels = normal_label + botnet_label

    if norm_and_standardize:
        temp = Summarizer()
        to_norm = ['avg_duration'] + list(temp.std_features.keys()) + list(temp.entropy_features.keys())
        normalize_features(header, to_norm, features)

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=.3, random_state=42)
    return np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)


def normalize_features(header, to_normalize, features):
    # Get all the features we want to normalize into a single array
    for feature in to_normalize:
        index = header.index(feature)
        values = [feature[index] for feature in features]
        fmin = min(values)
        fmax = max(values)
        for i in range(len(features)):
            try:
                features[i][index] = normalize(
                    features[i][index], fmin, fmax)
            except:
                features[i][index] = 0

        fmean = np.mean(values)
        fstd = np.std(values)
        for i in range(len(features)):
            features[i].append(standardize(
                features[i][index], fmean, fstd))


def to_tf_labels(labels):
    tflabels = []
    for label in labels:
        entry = [0 for _ in range(8)]
        entry[label] = 1
        tflabels.append(entry)
    return tflabels


def to_normal(labels):
    normals = []
    for label in labels:
        normals.append(label.index(1))
    return normals


def dl_train(features, label, use_bots=False, use_big_model=False,
             use_class_weight=False):
    model = Sequential()
    # TODO: Try random initialization.
    # TODO: Try using class weights for keras.
    model.add(
        Dense(64,
              input_dim=len(features[0]),
              kernel_initializer='random_uniform',
              activation='relu'))
    if use_big_model:
        # Simple model
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
    else:
        # Best model
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(246, activation='relu'))
        model.add(Dense(246, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))

    # TODO: Add labels for bot detection.
    if use_bots:
        model.add(Dense(8, activation='sigmoid'))
    else:
        model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                       metrics=['accuracy'])

    if use_bots:
        label = np.reshape(label, (-1, 8))
    else:
        label = np.reshape(label, (-1, 1))

    weights = None
    if use_class_weight:
        weights = class_weight.compute_class_weight('balanced',
                                                    [0, 1], label.flatten())
    model.fit(features, label, epochs=10, batch_size=32, verbose=False,
              class_weight=weights)
    return model


def dl_test(model, features, label, use_bots=False):
    if use_bots:
        label = to_normal(label)
    predicted = model.predict_classes(features, verbose=False)
    return (accuracy_score(label, predicted),
            precision_score(
                label, predicted,
                average='binary' if not use_bots else 'weighted'),
            recall_score(
                label, predicted,
                average='binary' if not use_bots else 'weighted'),
            f1_score(
                label, predicted,
                average='binary' if not use_bots else 'weighted'))


def dl_test_dict(model, features, label):
    predicted = model.predict_classes(features, verbose=False)
    metrics = ['accuracy', 'recall', 'precision', 'f1_score',
               'confusion_matrix']
    return dict(zip(metrics, (accuracy_score(label, predicted),
                              precision_score(label, predicted),
                              recall_score(label, predicted),
                              f1_score(label, predicted),
                              confusion_matrix(label, predicted))))


def dl_test_proba(model, features, label, normal_thresh=.5):
    probas = model.predict_proba(features, verbose=False)
    predicted = [(1 if prob[0] >= normal_thresh else 0) for prob in probas]
    metrics = ['accuracy', 'recall', 'precision', 'f1_score',
               'confusion_matrix']
    return dict(zip(metrics, (accuracy_score(label, predicted),
                              precision_score(label, predicted),
                              recall_score(label, predicted),
                              f1_score(label, predicted),
                              confusion_matrix(label, predicted))))


def test_proba(clf, features, label, normal_thresh=.5):
    probas = clf.predict_proba(features)
    predicted = [(0 if prob[0] >= normal_thresh else 1) for prob in probas]
    metrics = ['accuracy', 'recall', 'precision', 'f1_score',
               'confusion_matrix']
    return dict(zip(metrics, (accuracy_score(label, predicted),
                              precision_score(label, predicted),
                              recall_score(label, predicted),
                              f1_score(label, predicted),
                              confusion_matrix(label, predicted))))


def rf_train(features, label, use_attack=False, use_ahead=False, trees=50, max_features='auto', class_weight=None):
    # TODO: Use more trees for background.
    if use_attack or use_ahead:
        clf = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=trees), n_jobs=3)
        # bn = MultiLabelBinarizer()
        clf.fit(features, label)
    else:
        clf = RandomForestClassifier(
            class_weight=class_weight,
            max_features=max_features,
            # class_weight={0: 1, 1: 100},
            n_estimators=trees, n_jobs=4)  # n_estimators=700)
        clf.fit(features, label)
    return clf


def rf_compare_estimator_counts(xtrain, xtest, ytrain, ytest):
    estimator_counts = [10, 50, 100, 200, 300, 500, 700]
    scores = []
    for estimator in estimator_counts:
        clf = RandomForestClassifier(
            # class_weight={ 0: 1, 1: 6 },
            n_estimators=estimator,
            n_jobs=4)
        clf.fit(xtrain, ytrain)
        _, _, _, f1_score = test(clf, xtest, ytest)
        scores.append(f1_score)
    return scores


def dt_train(features, label):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, label)
    return clf


def test(clf, features, label, use_bots=False, use_attack=False,
         use_ahead=False):
    if use_attack:
        predicted = clf.predict(features)
        predicted_proba = clf.predict_proba(features)
        recall = {}
        precision = {}
        accuracy = {}
        f1_scores = {'micro': 0}
        attacks = ['Normal', 'ddos', 'spam', 'irc']
        for i in range(4):
            precision[i], recall[i], _ = precision_recall_curve(
                label[:, i],
                predicted_proba[:, i])
            f1_scores[i] = f1_score(label[:, i], predicted[:, i])

            accuracy[i] = average_precision_score(
                label[:, i], predicted_proba[:, i])
            print('{}: {}, {}, {}, {}'.format(
                attacks[i],
                accuracy[i],
                np.average(precision[i]),
                np.average(recall[i]),
                f1_scores[i]
            ))

        # Micro stands for the overall score for all classes
        precision['micro'], recall['micro'], _ = precision_recall_curve(
            label.ravel(), predicted_proba.ravel())
        f1_scores['micro'] = f1_score(label.ravel(), predicted.ravel())

        # precision['micro'] = ', '.join([str(x) for x in precision['micro']])
        # recall['micro'] = ', '.join([str(x) for x in recall['micro']])
        accuracy['micro'] = average_precision_score(
            label, predicted, average='micro')

        plot_multilabel_roc(precision, recall, 'ROC of attacks')
        return (accuracy, precision, recall, f1_scores)
    elif use_ahead:
        predicted = clf.predict(features)
        return (accuracy_score(label, predicted),
                precision_score(label, predicted, average='weighted'),
                recall_score(label, predicted, average='weighted'),
                f1_score(label, predicted, average='weighted'))
    else:
        predicted = clf.predict(features)
    return (accuracy_score(label, predicted),
            precision_score(
                label, predicted,
                average='binary' if not use_bots else 'weighted'),
            recall_score(
                label, predicted,
                average='binary' if not use_bots else 'weighted'),
            f1_score(
                label, predicted,
                average='binary' if not use_bots else 'weighted'))


def test_dict(clf, features, label, use_ahead=False):
    predicted = clf.predict(features)
    metrics = ['accuracy', 'recall', 'precision', 'f1_score',
               'confusion_matrix']
    return dict(zip(
        metrics, (accuracy_score(label, predicted),
                  precision_score(
                      label, predicted,
                      average='binary' if not use_ahead else 'weighted'),
                  recall_score(
                      label, predicted,
                      average='binary' if not use_ahead else 'weighted'),
                  f1_score(
                      label, predicted,
                      average='binary' if not use_ahead else 'weighted'),
                  confusion_matrix(label, predicted))))


def summary_of_detection(filename, model, use_bots=False, use_attack=False,
                         sample=False, use_ahead=False, steps_ahead=1, trees=50,
                         norm_and_standardize=False):
    """Genral call to train and test any model under any features."""
    if use_ahead:
        xtrain, xtest, ytrain, ytest = get_ahead_feature_labels(
            filename, Summarizer().features, steps_ahead)
    else:
        xtrain, xtest, ytrain, ytest = get_specific_features_from(
            filename, Summarizer().features, use_bots, use_attack,
            sample=sample, norm_and_standardize=norm_and_standardize)
    if model == 'rf':
        clf = rf_train(xtrain, ytrain, use_attack, use_ahead, trees=trees)
    elif model == 'dt':
        clf = dt_train(xtrain, ytrain)
    elif model == 'dl':
        if use_bots:
            ytrain = to_tf_labels(ytrain)
            ytest = to_tf_labels(ytest)
        clf = dl_train(xtrain, ytrain, use_bots)
        return dl_test(clf, xtrain, ytrain, use_bots)
    if use_attack:
        return [x['micro'] for x in test(
            clf, xtest, ytest, use_bots, use_attack)]
    return test(clf, xtest, ytest, use_bots, use_ahead=use_ahead)


def get_plots_for_each_interval(attack_type, intervals, model='rf'):
    """clf is a Random Forest model to test this all on."""
    scores = []
    for interval in intervals:
        filename = 'minute_aggregated/{}-{}s.featureset.csv'.format(
            attack_type, interval)
        try:
            _, _, _, f1_score = summary_of_detection(filename, model, trees=10)
        except:
            filename = 'minute_aggregated/{}-{}s.featureset.csv'.format(
                attack_type, int(interval))
            _, _, _, f1_score = summary_of_detection(filename, model, trees=10)

        scores.append(f1_score)
    return scores


def train_and_test_on(feature, label):
    xtrain, xtest, ytrain, ytest = train_test_split(
        feature, label, test_size=.3, random_state=42)
    model = rf_train(xtrain, ytrain)
    return test(model, xtest, ytest)
