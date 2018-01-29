"""Use machine learning on the new files.

TODO: Make it look for the right files. The rest should still be re-usable.

Good features so far:
    std_time
    src_to_dst
    entropy of all ports

Use for ROC curves:
    proba = clf.predict_proba(test_features).
    precision, recall, pr_threshold = precision_recall_curve(test_labels,
                                                             proba[:, 1])
    fpr, tpr, _ = roc_curve(test_labels, proba)  # proba[:, 1] for sklearn
    # roc_auc goes in the title
    roc_auc = auc(fpr, tpr)

    # Plot fpr vs tpr
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, \
    f1_score, precision_recall_curve, roc_curve, auc
import os


def get_files(directory):
    files = os.listdir(directory)
    return ['{}/{}'.format(directory, name) for name in files]


def get_roc_metrics(clf, features, labels):
    proba = clf.predict_proba(features)
    precision, recall, pr_threshold = precision_recall_curve(labels,
                                                             proba[:, 1])
    fpr, tpr, _ = roc_curve(labels, proba)  # proba[:, 1] for sklearn

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
    return features, labels


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
    return features, labels


def dl_trian(features, label):
    model = Sequential()
    model.add(
        Dense(64,
              input_dim=len(features[0]),
              kernel_initializer='uniform',
              activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                       metrics=['accuracy'])
    return model


def dl_test(model, features, label):
    predicted = model.predict_classes(features)
    return (accuracy_score(label, predicted),
            recall_score(label, predicted),
            precision_score(label, predicted))


def rf_train(features, label):
    clf = RandomForestClassifier()
    clf.fit(features, label)
    return clf


def dt_train(features, label):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, label)
    return clf


def test(clf, features, label):
    predicted = clf.predict(features)
    return (accuracy_score(label, predicted),
            recall_score(label, predicted),
            precision_score(label, predicted))


def summary_of_detection(filename):
    print('starting', filename)
    feature, label = get_feature_labels(filename)
    xtrain, xtest, ytrain, ytest = train_test_split(
        feature, label, test_size=.3, random_state=42)
    model = rf_train(xtrain, ytrain)
    return test(model, xtest, ytest)


def train_and_test_on(feature, label):
    xtrain, xtest, ytrain, ytest = train_test_split(
        feature, label, test_size=.3, random_state=42)
    model = rf_train(xtrain, ytrain)
    return test(model, xtest, ytest)


if __name__ == '__main__':
    print('starting')
    # files = get_files('aggregated_pcap')

    # for f in files:
    #     print("Recall: {}, Precision: {}".format(*summary_of_detection(f)))
    # print(summary_of_detection('minute_aggregated/ddos.featureset.csv'))
