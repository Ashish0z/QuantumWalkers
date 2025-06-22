from sklearn.decomposition import PCA

def reduce_features(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_train), pca.transform(X_test), pca