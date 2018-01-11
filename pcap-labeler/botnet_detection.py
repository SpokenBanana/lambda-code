from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import os
import numpy

def get_files(directory):
    files = os.listdir(directory)
    return ['{}/{}'.format(directory, name) for name in files]


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


def rf_train(features, label):
    clf = RandomForestClassifier()
    clf.fit(features, label)
    return clf


def test(clf, features, label):
    predicted = clf.predict(features)
    return recall_score(label, predicted), precision_score(label, predicted)


def summary_of_detection(filename):
    print('starting', filename)
    feature, label = get_feature_labels(filename)
    xtrain, xtest, ytrain, ytest = train_test_split(feature, label,
            test_size=.3, random_state=42)
    model = rf_train(xtrain, ytrain)
    return test(model, xtest, ytest)



if __name__ == '__main__':
    files = get_files('aggregated_pcap')

    for f in files:
        print("Recall: {}, Precision: {}".format(*summary_of_detection(f)))

