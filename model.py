import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

class HARCSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

class CNN_LSTM_HAR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_LSTM_HAR, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc(output)
        return output

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss, correct = 0.0, 0
    for X, y in loader:
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader, loss_fn):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            loss = loss_fn(out, y)
            running_loss += loss.item() * X.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)
