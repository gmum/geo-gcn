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
