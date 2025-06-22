import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes=2):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_dnn(X_train, y_train, X_test, config):
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    num_classes = len(torch.unique(y_train))
    
    model = DNN(X_train.shape[1], config['hidden_size'], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    preds = model(X_test).argmax(dim=1).cpu().numpy()
    return model, preds
