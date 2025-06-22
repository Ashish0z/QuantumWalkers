import torch
import pennylane as qml
from pennylane import numpy as np

device = torch.device("cpu")

def qnn_model(n_qubits, wires):
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=wires)
        qml.BasicEntanglerLayers(weights, wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]
    return circuit

def train_qnn(X_train, y_train, X_test, config):
    n_qubits = config['n_qubits']
    dev = qml.device("default.qubit", wires=n_qubits)


    qlayer = qml.qnn.TorchLayer(qml.QNode(qnn_model(n_qubits, range(n_qubits)), dev, interface="torch"), {"weights": (config['entangling_layers'], n_qubits)})

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    n_classes = len(torch.unique(y_train))  # Dynamically determine output size

    model = torch.nn.Sequential(
        qlayer,
        torch.nn.Linear(n_qubits, n_classes)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for _ in range(config['epochs']):
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()

    preds = model(X_test).argmax(dim=1).cpu().numpy()
    return model, preds