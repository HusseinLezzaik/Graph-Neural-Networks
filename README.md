# Graph Neural Networks - Classification/Regression

## Introduction
Graphs express entities (nodes) along with their relations (edges), and both nodes and edges can be typed (e.g., `"user"` and `"item"` are two different types of nodes). DGL provides a graph-centric programming abstraction with its core data structure – `DGLGraph`. `DGLGraph` provides its interface to handle a graph’s structure, its node/edge features, and the resulting computations that can be performed using these components.

## Overview of the Repository
In this repo, you'll find code for:

* `Build_GNN_module.py`: code for DGL’s message passing APIs, and GraphSAGE convolution module.
* `DGL_Graph_Construction.py`: code for graph construction from scratch.
* `Link_Prediction_Using_GNNs.py`: code for link prediction using dgl.
* `Make_your_own_dataset.py`: code for creating a cataset for node classification or link prediction from csv.
* `Node_Classification_with_DGL.py`: code for building a GNN for semi-supervised node classification.
* `Training_GNN_for_Graph_Classfication.py`: code for training gnn model for graph classification.


## Getting Started
1.  Clone repo: `git clone https://github.com/HusseinLezzaik/Graph-Neural-Networks.git`
2.  Install dependencies:
    ```
    conda create -n graph-nets python=3.7
    conda activate graph-nets
    pip install -r requirements.txt
    ```
3.  You also need to download the Deep Graph Library (DGL) from [here](https://docs.dgl.ai/install/index.html). It works fine with Ubunut 20.04.

## Acknowledgements
Adapted from the official DGL documentation [here](https://docs.dgl.ai/tutorials/blitz/index.html).

## Contact
* Hussein Lezzaik : hussein dot lezzaik at gmail dot com
