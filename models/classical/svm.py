from sklearn.svm import SVC

def train_svm(X_train, y_train, X_test, config):
    clf = SVC(kernel='rbf', gamma='scale', C=config['C'])
    clf.fit(X_train, y_train)
    return clf, clf.predict(X_test)