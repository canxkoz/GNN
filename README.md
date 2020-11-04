# GNN
GCN/GAT implementations using PyToch Geometric

Important Info:
## Task 1
```gcn-cora.py``` is a standalone file which just downloads and trains the GCN/GAT on the Cora Dataset.

## Task 2
- [PATTERN](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/SBMs/generate_SBM_PATTERN.ipynb) (Works perfectly.)
- [CLUSTER](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/SBMs/generate_SBM_CLUSTER.ipynb) (This one is a bit too randomised gives bad results.)
The two Jupyter Notebooks above creates randomized graphs consisting of numbers. Both of these Jupyter Notebooks will create 3 plk files which will consist of datasets for train, test, validation. 

Essentially what these will yiled is a large list of small sized graphs relative to Cora. There is a graph at each index of a list. We have to find a way to convert this long list of graphs to the type that pytorch_geometric can read. 

It would be ideal if you used a GPU to train these because each epoch takes 6 minutes using a CPU and you need 400 Epochs.

The two files below work together
```sbm_data2geometric.py```: Converts synthatically generated dataset to a format that pytorch_geometric can read. 
```train.py``` : Uses the synthetically converted datasets and trains the GCN/GAT.

I have added comments on almost each line for both files. I hope that this accelerates your progress. 

