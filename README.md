# Spatial Graph Convolutional Networks

This repository contains an implementation of [Spatial Graph Convolutional Neural Networks (SGCN)](https://arxiv.org/abs/1909.05310).

# Dependencies

- PyTorch >= 1.1
- PyTorch geometric >= 1.1.2

# Running the code

To run geo-GCN on MNISTSuperpixels with default parameters, go to `src` and use the command:

```python
python train_models.py MNISTSuperpixels
```
 
 To use chemical data:
 
 ```python
from torch_geometric.data import DataLoader
from chem import load_dataset

batch_size = 64
dataset_name = ...  # 'freesolv' / 'esol' / 'bbbp'

train_dataset = load_dataset(dataset_name, 'train')
val_dataset = load_dataset(dataset_name, 'val')
test_dataset = load_dataset(dataset_name, 'test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# training loop
... 
```

## Other options

The code allows to manipulate some of the parameters (for example using other versions of the model, changing learning rate values or optimizer types). For more information, see the list of available arguments in `src/train_models.py` file.

## Reference

If you make use of our results or code in your research, please cite the following:

```
@InProceedings{
10.1007/978-3-030-63823-8_76,
author="Danel, Tomasz
and Spurek, Przemys{\l}aw
and Tabor, Jacek
and {\'{S}}mieja, Marek
and Struski, {\L}ukasz
and S{\l}owik, Agnieszka
and Maziarka, {\L}ukasz",
editor="Yang, Haiqin
and Pasupa, Kitsuchart
and Leung, Andrew Chi-Sing
and Kwok, James T.
and Chan, Jonathan H.
and King, Irwin",
title="Spatial Graph Convolutional Networks",
booktitle="Neural Information Processing",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="668--675",
abstract="Graph Convolutional Networks (GCNs) have recently become the primary choice for learning from graph-structured data, superseding hash fingerprints in representing chemical compounds. However, GCNs lack the ability to take into account the ordering of node neighbors, even when there is a geometric interpretation of the graph vertices that provides an order based on their spatial positions. To remedy this issue, we propose Spatial Graph Convolutional Network (SGCN) which uses spatial features to efficiently learn from graphs that can be naturally located in space. Our contribution is threefold: we propose a GCN-inspired architecture which (i) leverages node positions, (ii) is a proper generalization of both GCNs and Convolutional Neural Networks (CNNs), (iii) benefits from augmentation which further improves the performance and assures invariance with respect to the desired properties. Empirically, SGCN outperforms state-of-the-art graph-based methods on image classification and chemical tasks.",
isbn="978-3-030-63823-8"
}
```

