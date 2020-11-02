import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

from torch_geometric.datasets import Planetoid
# Get dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

USE_MODEL = "GAT"

class GCN(torch.nn.Module):
    def __init__(self):
        '''
        Input layer num_node_features = 1433
        Hidden layer node = 4*16
        Hidden layer node = 4*16
        Out layer num_classes = 7
        '''
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=dataset.num_node_features, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=16)
        self.conv3 = GCNConv(in_channels=16, out_channels=dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        '''
        Input layer num_node_features = 1433
        Hidden layer node = 4*16
        Hidden layer node = 4*16
        Out layer num_classes = 7
        '''
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=dataset.num_node_features, out_channels=16, heads=4, dropout=0.2)
        self.conv2 = GATConv(in_channels=4*16, out_channels=16, heads=4, dropout=0.2)
        self.conv3 = GATConv(in_channels=4*16, out_channels=dataset.num_classes, heads=6, concat=False, dropout=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    print()
    print("Dataset = Cora")
    print("Number of graphs in CORA =", len(dataset))
    print("Number of nodes in graph[0] =", dataset[0].x.shape[0])
    print("Number of node features =", dataset.num_node_features)
    print("Number of node classes =", dataset.num_classes)
    print("Number of edges in graph[0] =", dataset[0].edge_index.shape[1]//2)
    print("Number of edge features =", dataset.num_edge_features)
    print()

    # Use CPU
    device = torch.device('cpu')
    # Transfer model params to choosen device
    if USE_MODEL == "GAT":
        model = GAT().to(device)
    else:
        model = GCN().to(device)

    # Transfer dataset to choosen device
    data = dataset[0].to(device)
    # Create optimizer
    # lr = learning rate 
    # weight_decay = weight decay rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    TOTAL_EPOCH = 400
    for epoch in range(TOTAL_EPOCH):
        # Switch model to train mode
        model.train()

        # Reset gradients
        optimizer.zero_grad()
        # Forward pass(prediction)
        out = model(data)
        # Get loss use y(prediction) and y_truth(labels)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # Find gradients
        train_loss.backward()
        # Optimize weight params
        optimizer.step()

        # Calc validation loss at every 20 epoch
        if epoch % 20 == 0:

            # Switch model to evaluation mode
            model.eval()

            # EXTRA STEPS TO GET TRAIN ACC
            # Make prediction
            _, pred = model(data).max(dim=1)
            # Get number of 'True' predictions for test data
            correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
            # Find acc
            # ACC = Number of 'True' / Number of Total
            train_acc = correct / int(data.train_mask.sum())

            # Wont compute gradients over validation set
            with torch.no_grad():
                # Make prediction
                _, pred = model(data).max(dim=1)
                # Get number of 'True' predictions for validation data
                correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                # Find acc
                # ACC = Number of 'True' / Number of Total
                val_acc = correct / int(data.val_mask.sum())

                # EXTRA STEPS TO GET VALIDATION LOSS
                # Forward pass(prediction)
                out = model(data)
                # Get loss use y(prediction) and y_truth(labels)
                val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])

                print('Train Accuracy: {:.4f} Validation Accuracy: {:.4f} Train Loss: {:.4f} Validation Loss: {:.4f} Epoch: {}/{}'.format(train_acc, val_acc, train_loss, val_loss, epoch, TOTAL_EPOCH))


    # Switch model to evaluation mode
    model.eval()
    # Make prediction
    _, pred = model(data).max(dim=1)
    # Get number of 'True' predictions for test data
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    # Find acc
    # ACC = Number of 'True' / Number of Total
    acc = correct / int(data.test_mask.sum())
    # Print
    print('Test Accuracy: {:.4f}'.format(acc))
