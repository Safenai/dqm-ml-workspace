# DQM-ML PyTorch

An extension for DQM-ML V2 providing advanced metrics that require PyTorch and deep learning models.

## Features

* Domain Gap Analysis: Measures statistical distance between source and target distributions using:
  * `fid` (Frechet Inception Distance)
  * `klmvn_diag` (KL-Divergence)
  * `mmd_linear` (Maximum Mean Discrepancy)
  * `wasserstein_1d`

## Requirements

* `torch`
* `torchvision`
* `scipy`
