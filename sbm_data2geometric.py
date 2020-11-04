import time
import torch
import scipy
import pickle
from tqdm import tqdm
from torch_geometric import utils
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        """
            Loading SBM datasets CLUSTER-PATTERN
            https://github.com/graphdeeplearning/benchmarking-gnns/blob/bdb9f6817f7e26a5e7dddc865e2e9e82bc59faa2/data/SBMs.py#L149
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (path))
        self.path = path
        with open(path+'_train.pkl',"rb") as f:
            f = pickle.load(f)
            train = f
            # To able to use data in PyTorch Geometric we have to transform it
            self.train_dataset = self.sbm_2_geometric(train)
        with open(path+'_test.pkl',"rb") as f:
            f = pickle.load(f)
            test = f
            # To able to use data in PyTorch Geometric we have to transform it
            self.test_dataset = self.sbm_2_geometric(test)
        with open(path+'_val.pkl',"rb") as f:
            f = pickle.load(f)
            val = f
            # To able to use data in PyTorch Geometric we have to transform it
            self.val_dataset = self.sbm_2_geometric(val)
        print("Train dataset")
        print(self.train_dataset[0])
        print("Test dataset")
        print(self.test_dataset[0])
        print("Val dataset")
        print(self.val_dataset[0])
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def sbm_2_geometric(self, data):
        '''
            Transform sbm data to geometric data
        '''

        # PyTorch Geometric Dataset
        dataset = []

        # Traverse graphs
        for graph in tqdm(data):
            # ##GET EDGE MATRIX
            
            W = graph['W']
            W = scipy.sparse.csr_matrix(W)
            edge_index, edge_weight = utils.from_scipy_sparse_matrix(W)

            # ##GET NODE MATRIX
        
            # Node feature matrix in PyTorch Geometric format
            x = []
            features = graph['node_feat']
            for feature in features:
                x.append([feature])
            
            # x must be torch tensor and float(Conv layer returns error if not)
            x = torch.tensor(x).float()

            # ##GET Y(labels)

            # Node labels matrix in PyTorch Geometric format (Same format)
            # y must be long because loss functions want it
            y = graph['node_label'].long()

            # Create PyTorch geometric graph
            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)

        return dataset

