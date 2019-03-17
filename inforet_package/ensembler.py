import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


def fit_multiple_classifiers(classifiers, X_list, y):
    assert len(classifiers) == len(X_list)
    return [clf.fit(X, y) for clf, X in zip([clf for _, clf in classifiers], X_list)]


def average_voting(fitted_classifiers, X_list):
    probabilities = np.asarray([clf.predict_proba(X) for clf, X in zip(fitted_classifiers, X_list)])
    avg = np.average(probabilities, axis=0)
    return np.argmax(avg, axis=1)


def split_dataset(X_list, y):
    n = X_list[0].shape[0]
    indices = np.random.permutation(n)
    A_idx, B_idx = indices[:n//2], indices[n//2:]
    X_a, X_b = [X[A_idx, :] for X in X_list], [X[B_idx, :] for X in X_list]
    y_a, y_b = y[A_idx], y[B_idx]
    return X_a, X_b, y_a, y_b


def split_train_test_list(X_list, y, test_size=0.3):
    idx = np.arange(0, X_list[0].shape[0])
    X_train = []
    X_test = []
    _, _, y_train, y_test, idx_train, idx_test = train_test_split(X_list[0], y, idx, test_size=test_size)
    for x in X_list:
        X_train.append(x[idx_train])
        X_test.append(x[idx_test])

    return X_train, X_test, y_train, y_test


class StackingEnsembler:
    def __init__(self, classifiers, meta_learner):
        self.classifiers = classifiers
        self.fitted_classifiers = None
        self.meta_learner = meta_learner

    def fit(self, X_list, y):
        X_a, X_b, y_a, y_b = split_dataset(X_list, y)
        self.fitted_classifiers = fit_multiple_classifiers(self.classifiers, X_a, y_a)

        pred = np.asarray([clf.predict(X) for clf, X in zip(self.fitted_classifiers, X_b)])
        self.meta_learner.fit(pred.T, y_b)

    def predict(self, X_list):
        first_level_pred = np.asarray([clf.predict(X) for clf, X in zip(self.fitted_classifiers, X_list)])
        return self.meta_learner.predict(first_level_pred.T)


def train_predict_svm(X, y, C=1, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    clf = svm.SVC(kernel='linear', C=C, max_iter=-1, probability=True, verbose=5)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), y_test
