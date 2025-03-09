import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# simple MLP architecture, hidden layer with 100 neurons
class MLP_model(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        return self.model(x)
    
# let the training beginnnnnn
def train():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# let the testing beginnnnnn
def test():
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_mse = criterion(y_pred_test, y_test).item()
    print(f"Test RMSE: {np.sqrt(test_mse)}")

if __name__ == "__main__":

    # load data
    data = pd.read_csv("data/preproc_mixtures.csv")

    # split into inputs (fluids) and outputs (peaks)
    X = data.iloc[:, :6].values
    y = data.iloc[:, 6:].values

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # pytorch tensor bullshit
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # pytorch dataloader bullshit
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # initialise model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLP_model(input_size, output_size)

    # some hyperparams
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # train and test
    train()
    test()

    # empirical testing
    model.eval()
    inp = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out = model(inp).numpy()

    print(out >= 150)
    print("Predicted marker peak heights:", out)
    