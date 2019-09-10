# Geometric Graph Convolutional Neural Networks

This repository contains an implementation of [Geometric Graph Convolutional Neural Networks(geo-GCN)](https://www.google.com).

# Dependencies

- PyTorch >= 1.1
- PyTorch geometric >= 1.1.2

# Running the code

To run geo-GCN on MNISTSuperpixels with default parameters, go to `src` and use the command:

```python
python train_models.py MNISTSuperpixels
```
 

## Other options

The code allows to manipulate some of the parameters (for example using other versions of the model, changing learning rate values or optimizer types). For more information, see the list of available arguments in `src/train_models.py` file.
