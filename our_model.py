import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

# model parameters
input_dim = 128  # Dimension of input features x
hidden_channel = 512
num_categories = 40 # output channels to the model
p_dropout = 0.25
num_epochs = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
dataset = torch.load('data/hw3/processed/data.pt')
# dataset = HW3Dataset(root='data/hw3/')
# dataset = dataset[0]

# split data to: nodes, edges, labels, train & val set
edge_index = dataset.edge_index.to(device)
x = dataset.x.to(device)
y = dataset.y.to(device)
train_mask = dataset.train_mask.to(device)
val_mask = dataset.val_mask.to(device)

class GNN(torch.nn.Module):
    def __init__(self,input_dim, hidden_channel, num_categories):
        super(GNN, self).__init__()
        torch.manual_seed(12)
        self.conv1 = GCNConv(input_dim, hidden_channel).to(device)
        self.conv2 = GCNConv(hidden_channel, num_categories).to(device)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=p_dropout)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, criterion):

    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x, edge_index)
        loss = criterion(output[train_mask],y[train_mask].squeeze())
        loss.backward()
        optimizer.step()
        print(f"Epoch number: {epoch}, Training Loss: {loss.item():.3f}")


def val(model):

    model.eval()

    with torch.no_grad():
        output_test = model(x, edge_index)[val_mask]
        _, pred_labels = output_test.max(dim=1)

    return pred_labels.cpu()


def accuracy_calc(real_labels, pred_labels):
    success = 0
    for real, pred in zip(real_labels, pred_labels):
        if real == pred:
            success += 1
    total_labels = real_labels.size(0)
    return (success / total_labels) * 100


def run_model(input_dim, hidden_channel, num_categories):

    model = GNN(input_dim, hidden_channel, num_categories).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, optimizer, criterion)
    pred_labels = val(model)

    # save trained model
    torch.save(model.state_dict(), "trained_model.pt")

    real_labels = y[val_mask].squeeze().cpu()
    accuracy = accuracy_calc(real_labels, pred_labels)
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    run_model(input_dim, hidden_channel, num_categories)





