# Online Feature Selection with Varying Feature Spaces

![Python 3.6](https://img.shields.io/badge/python-3.8-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Abstract
Feature selection, an essential technique in data mining, is often confined to batch learning or online idealization of data scenarios despite its significance. Existing online feature selection methods have specific assumptions regarding data stream, such as requiring a fixed feature space with an explicit pattern and complete labelling of samples. Unfortunately, data streams generated in many real scenarios commonly exhibit arbitrarily incomplete feature spaces and scarcity labels, making existing approaches be unsuitable for real applications. To fill these gaps, this study proposes a new problem called Online Feature Selection with Varying Features Spaces (OFSVF). OFSVF has a threefold main idea: 1) it leverages Gaussian Copula to model the incomplete feature correlation in a complete latent space, encoded by continuous variables, 2) it employs a novel tree-ensemblebased approach to select the most informative features on-thefly, and 3) it develops the underlying geometric structure of instances to establish the relationship between unlabeled and labels. Experimental results are documented to demonstrate the feasibility and effectiveness of our proposed method.

## File

The overall framework of this project is designed as follows
1. The **dataset** file is used to hold the datasets and lables

2. The **source** file is all the code for the model

3. The **Result** is for saving relevant results (e.g. CER, Figure)

### Getting Started


1. Make sure you meet package requirements by running:

```python
pip install -r requirements.txt
```

2. Running OFSVF model

```python
python OFSVF_VFS.py
```
