from dataset import HW3Dataset
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# load data
dataset = HW3Dataset(root='data/hw3/')
dataset = dataset[0]
# model parameters
input_dim = 128  # Dimension of input features x
hidden_channel = 512
num_categories = 40 # output channels to the model
p_dropout = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

x = dataset.x.to(device)
edge_index = dataset.edge_index.to(device)

# load trained model
trained_model = GNN(input_dim, hidden_channel, num_categories)
trained_model.load_state_dict(torch.load("trained_model.pt"))

# predict categories
trained_model.eval()
with torch.no_grad():
    output_test = trained_model(x, edge_index)
    _, pred_labels = output_test.max(dim=1)

# save results as csv file
with open("prediction.csv", "w") as f:
    f.write("idx,prediction\n")
    for i in range(len(pred_labels)):
        f.write(f"{int(i)},{int(pred_labels[i])}\n")

