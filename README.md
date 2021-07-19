# PyTorch version of Deep Learning for Asset Pricing

# Introduction
This repository contains PyTorch implementation of the asset pricing model from the paper [Deep Learning in Asset Pricing](https://arxiv.org/pdf/1904.00745.pdf) by Luyang Chen, Markus Pelger, and Jason Zhu (September 10, 2020).

An official TensorFlow implementation from [repo](https://github.com/LouisChen1992/Deep_Learning_Asset_Pricing) was used as a prototype for this implementation.

# Setup
Dependencies for Python 3.7:
- Numpy 1.21.1 
- PyTorch 1.9.0

Install the required packages
```
pip install -r requirements.txt
```
# Usage
Download the required dataset from the [link](https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l).
Place unpacked folder ```datasets``` in the main folder of the project.
One could use command line to download the dataset using ```gdown``` package:
```
pip install gdown
gdown "https://drive.google.com/uc?id=1h9O7YwPLaRBbghtF50Cr-JmIq0aHHi4Y"
unzip datasets.zip
```
Run the script to train both GAN and Returns prediction models:
```
python run_torch.py
```

The list of arguments for the script is available by:
```
python run_torch.py --help
```

# Results
Script stores model dumps and logging file in the folder with path given by ```--path_to_output``` option.
In the end of the logging file one could find calculated statistics (Explained Variation, XS-R2, Weighted XS-R2) for each dataset.
To calculate statistics on the given datasets and pretrained model run the following script:
```
python calculate_statistics.py
```
The results from this implementation are the following:

| Dataset | Explained Variation | XS-R2 | Weighted XS-R2 |
| --- | ----------- | -----| ------ |
| Train | 0.089 | 0.035 | 0.154 |
| Validation | 0.011 | -0.004 | -0.015 |
| Test | 0.015 | 0.004 | 0.032 |



# GPU support
The script automatically checks if there is a GPU available.
One could check if GPU is available and correctly installed using:
```
import torch
if torch.cuda.is_available():
    print("Cuda is available")
```


