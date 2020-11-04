# GNN
GCN/GAT implementations using PyToch Geometric

Important Info:

gcn-cora.py is a standalone file which just downloads and trains the GCN/GAT on the Cora Dataset.

The two files below work together
sbm_data2geometric.py : Converts synthatically generated dataset (the code for the synthetic generation of the datasets can be found here.)
https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/SBMs/generate_SBM_PATTERN.ipynb
https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/SBMs/generate_SBM_CLUSTER.ipynb (This one is a bit too randomised gives bad results)
train.py : Uses the synthetically converted datasets and trains the GCN/GAT.

I have added comments on almost each line for both files. I hope that this accelerates your progress. 

