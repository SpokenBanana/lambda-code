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
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, \
    f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix
import os
from absl import app
from absl import flags
from utils import best_features
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('attack_type', None, 'Type of attack to train on.')
flags.DEFINE_string('model_type', None, 'Type of model to train with.')
flags.DEFINE_integer('interval', None, 'Interval of the file to train on.')


def get_files(directory):
    files = os.listdir(directory)
    return ['{}/{}'.format(directory, name) for name in files]


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


def get_specific_features_from(filename, feature_names=None):
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
            labels.append(1 if info[-1] == 'Botnet' else 0)
    xtrain, xtest, ytrain, ytest = train_test_split(
        features, labels, test_size=.3, random_state=42)
    return xtrain, xtest, ytrain, ytest


def dl_train(features, label):
    model = Sequential()
    model.add(
        Dense(64,
              input_dim=len(features[0]),
              kernel_initializer='uniform',
              activation='relu'))
    # Added hidden layers.
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(246, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                       metrics=['accuracy'])
    label = np.reshape(label, (-1, 1))
    model.fit(features, label, epochs=10, batch_size=32, verbose=False)
    return model


def dl_test(model, features, label):
    predicted = model.predict_classes(features, verbose=False)
    return (accuracy_score(label, predicted),
            recall_score(label, predicted),
            precision_score(label, predicted),
            f1_score(label, predicted))


def dl_test_dict(model, features, label):
    predicted = model.predict_classes(features, verbose=False)
    metrics = ['accuracy', 'recall', 'precision', 'f1_score',
               'confusion_matrix']
    return dict(zip(metrics, (accuracy_score(label, predicted),
                              recall_score(label, predicted),
                              precision_score(label, predicted),
                              f1_score(label, predicted),
                              confusion_matrix(label, predicted))))


def rf_train(features, label):
    clf = RandomForestClassifier(n_estimators=700)
    clf.fit(features, label)
    return clf


def rf_compare_estimator_counts(xtrain, xtest, ytrain, ytest):
    estimator_counts = [50, 100, 200, 300, 500, 700, 800, 900, 1000, 1200]
    scores = []
    for estimator in estimator_counts:
        clf = RandomForestClassifier(n_estimators=estimator)
        clf.fit(xtrain, ytrain)
        _, _, _, f1_score = test(clf, xtest, ytest)
        scores.append(f1_score)
    return scores


def dt_train(features, label):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, label)
    return clf


def test(clf, features, label):
    predicted = clf.predict(features)
    return (accuracy_score(label, predicted),
            recall_score(label, predicted),
            precision_score(label, predicted),
            f1_score(label, predicted))


def test_dict(clf, features, label):
    predicted = clf.predict(features)
    metrics = ['accuracy', 'recall', 'precision', 'f1_score',
               'confusion_matrix']
    return dict(zip(metrics, (accuracy_score(label, predicted),
                              recall_score(label, predicted),
                              precision_score(label, predicted),
                              f1_score(label, predicted),
                              confusion_matrix(label, predicted))))


def summary_of_detection(filename, model):
    xtrain, xtest, ytrain, ytest = get_specific_features_from(
        filename, best_features())
    if model == 'rf':
        clf = rf_train(xtrain, ytrain)
    elif model == 'dt':
        clf = dt_train(xtrain, ytrain)
    elif model == 'dl':
        clf = dl_train(xtrain, ytrain)
        return dl_test(clf, xtrain, ytrain)
    return test(clf, xtest, ytest)


def get_plots_for_each_interval(attack_type):
    """clf is a Random Forest model to test this all on."""
    intervals = [1, 3, 5, 10, 20, 30, 60, 120, 180]
    scores = []
    for interval in intervals:
        filename = 'minute_aggregated/{}-{}s.featureset.csv'.format(
            attack_type, interval)
        _, _, _, f1_score = summary_of_detection(filename, 'rf')
        scores.append(f1_score)
    return scores


def train_and_test_on(feature, label):
    xtrain, xtest, ytrain, ytest = train_test_split(
        feature, label, test_size=.3, random_state=42)
    model = rf_train(xtrain, ytrain)
    return test(model, xtest, ytest)


def main(_):
    base_name = 'minute_aggregated/{}-{}s.featureset.csv'
    f = base_name.format(FLAGS.attack_type, FLAGS.interval)
    # print(device_lib.list_local_devices())
    print("Accuracy: {}, Recall: {}, Precision: {}, f1_score: {}".format(
        *summary_of_detection(f, FLAGS.model_type)))


if __name__ == '__main__':
    app.run(main)
