# Graph Neural Networks - Classification/Regression

## Introduction
Graphs express entities (nodes) along with their relations (edges), and both nodes and edges can be typed (e.g., `"user"` and `"item"` are two different types of nodes). DGL provides a graph-centric programming abstraction with its core data structure – `DGLGraph`. `DGLGraph` provides its interface to handle a graph’s structure, its node/edge features, and the resulting computations that can be performed using these components.

In this repo, you'll find code for:
* Node Classification with DGL
* Representing a Graph with DGL
* Building your own GNN module
* Link Prediction using GNNs
* Training a GNN for Graph Classification
* Making your own Graph Dataset 


## Requirements
1. Install PyTorch from [here](https://pytorch.org/)
2. Run the following command to install additional dependencies:
``` 
pip install -r requirements.txt 
```
You also need to download the Deep Graph Library (DGL) from [here](https://docs.dgl.ai/install/index.html). It works fine with Ubunut 20.04.

## Acknowledgements
Adapted from the official DGL documentation [here](https://docs.dgl.ai/tutorials/blitz/index.html).
