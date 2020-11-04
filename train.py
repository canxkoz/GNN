import time
import torch
import pickle
from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv

from sbm_data2geometric import SBMsDataset, DotDict

#DATASET = "data_gen/SBM_CLUSTER"
DATASET = "../data_gen/SBM_PATTERN"
USE_MODEL = "GCN"
#USE_MODEL = "GAT"
TOTAL_EPOCH = 400
LEARNING_RATE = 0.01
DEVICE = 'cpu'

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        '''
        Input layer num_node_features = num_node_features
        Hidden layer node = 4*16
        Hidden layer node = 4*16
        Out layer num_classes = num_classes
        '''
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=num_node_features, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=16)
        self.conv3 = GCNConv(in_channels=16, out_channels=num_classes)

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
    def __init__(self, num_node_features, num_classes):
        '''
        Input layer num_node_features = num_node_features
        Hidden layer node = 4*16
        Hidden layer node = 4*16
        Out layer num_classes = num_classes
        '''
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=16, heads=4, dropout=0.2)
        self.conv2 = GATConv(in_channels=4*16, out_channels=16, heads=4, dropout=0.2)
        self.conv3 = GATConv(in_channels=4*16, out_channels=num_classes, heads=6, concat=False, dropout=0.2)

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
    print("DATA SET PREPAIRING")
    print()    

    # Get SBM data in PyTorch Geometric format
    dataset = SBMsDataset(path=DATASET)

    # We need to use DataLoader to split data to minibatches
    train_loader = DataLoader(dataset.train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=64)
    test_loader = DataLoader(dataset.test_dataset, batch_size=64)

    # Get number of feature and number of classes to build a network
    # For one of the graphs in train dataset we get these results:

    # Data(edge_index=[2, 4324], x=[101, 1], y=[101])
    # This means there are multiple graphs in train or other sets(NOT LIKE CORA)
    # In this particular graph from train set there are 101 node and one node contains just 1 feature
    # In this particular graph there are 4324 edge. If this graph undirected there are 4324/2 edges total
    # Number of labels(y) must be same with number of nodes
    num_node_features = dataset.train_dataset[0]['x'].shape[1]
    num_classes = torch.unique(dataset.train_dataset[0]['y']).shape[0]

    print()
    print("TRAINING START")
    print()

    # Use CPU
    device = torch.device(DEVICE)

    # Transfer model params to choosen device
    if USE_MODEL == "GAT":
        model = GAT(num_node_features=num_node_features, num_classes=num_classes).to(device)
    else:
        model = GCN(num_node_features=num_node_features, num_classes=num_classes).to(device)

    # Loss function
    nll_loss = torch.nn.NLLLoss()

    # Acc function
    def acc(pred, truth):
        # Get number of 'True' predictions for test data
        correct_results_sum = (pred == truth).sum().float()
        # ACC = Number of 'True' / Number of Total
        acc = correct_results_sum/truth.shape[0]
        
        return acc

    # Create optimizer
    # lr = learning rate 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(TOTAL_EPOCH):
        # Keep time
        start_time = time.time()
        # Switch model to train mode
        model.train()
        # Reset gradients
        optimizer.zero_grad()
        # Keep total_train_loss. Will reset for every epoch
        total_train_loss = 0
        # Get a train batch to train
        for data in tqdm(train_loader):
            # Copy batch to device mem
            data = data.to(device)
            # Reset gradients
            model.zero_grad()
            # Forward Pass(Prediction)
            outputs = model(data)
            # Calculate loss
            loss = nll_loss(outputs, data.y)
            # Keep loss
            total_train_loss += loss.item()
            # Backward Pass to calculate the gradients
            loss.backward()
            # Optimize model params w/ gradients
            optimizer.step()
            
        # Calculate the average loss over all of the batches for current epoch.
        avg_train_loss = total_train_loss / len(train_loader)

        # Calculate validation loss and acc to inform user
        # Switch model to evaluation mode
        model.eval()
        # Wont compute gradients over validation set
        with torch.no_grad():
            # Keep total_validation_loss and total_validation_acc
            total_validation_loss = 0
            total_validation_acc = 0
            # Get a validation batch to calculate acc and loss
            for data in val_loader:
                # Copy batch to device mem
                data = data.to(device)
                # Forward Pass(Prediction)
                outputs = model(data)
                # Calculate loss
                loss = nll_loss(outputs, data.y)
                # Keep loss
                total_validation_loss += loss.item()

                # Calculate accuracy
                # We get the index of the maximum
                _, pred = outputs.max(dim=1)
                validation_acc = acc(pred, data.y)
                # Keep acc
                total_validation_acc += validation_acc

            # Calculate the average loss over all of the batches for current epoch.
            avg_validation_loss = total_validation_loss / len(val_loader)
            # Calculate the average acc over all of the batches for current epoch.
            avg_validation_acc = total_validation_acc / len(val_loader)

        end_time = time.time()
        duration = end_time - start_time        
        print('Validation Accuracy: {:.4f} Train Loss: {:.4f} Validation Loss: {:.4f} Duration {:.2f} Epoch: {}/{}'.format(avg_validation_acc, avg_train_loss, avg_validation_loss, duration, epoch+1, TOTAL_EPOCH))
