import pennylane as qml
from pennylane.kernels import square_kernel_matrix, kernel_matrix
from sklearn.svm import SVC
import torch

def train_q_svm(X_train, y_train, X_test):
    n_qubits = X_train.shape[1]

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(y, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    def q_kernel(x, y):
        return circuit(x, y)[0]

    K_train = square_kernel_matrix(X_train, q_kernel)
    K_test = kernel_matrix(X_test, X_train, q_kernel)

    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    preds = clf.predict(K_test)
    return clf, preds
