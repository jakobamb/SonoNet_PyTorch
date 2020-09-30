# PyTorch SonoNet

**Disclaimer**  
These files come without any warranty!  
In particular, there might be unforeseen differences to the original implementation.

### About this repository

This is a PyTorch implementation of SonoNet:

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This repository is based on https://github.com/baumgach/SonoNet-weights which provides a theano+lasagne implementation.

### Files
sononet/sononet.py:  
PyTorch implementation of the original models.py file.

sononet/SonoNet16.pth, sononet/SonoNet32.pth, sononet/SonoNet64.pth:  
The original pretrained weights converted into PyTorch format.

test.py:  
Modified version of the original example.py file. This file runs classification on the examples images.

### Dependencies
NumPy, Pillow, Matplotlib, PyTorch.  
Tested with PyTorch 0.4.0 and 1.3.1.

### Installing as Python module

After installing the dependencies, run
```
cd SonoNet_PyTorch
pip install .
```

### Usage
After installing the dependencies, classify the example example images with:
```
python SonoNet_Pytorch/test.py
```

After installing as Python module (see above), import SonoNet with:
```
from sononet import SonoNet
```
