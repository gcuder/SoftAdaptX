[![Tests](https://img.shields.io/github/actions/workflow/status/gcuder/SoftAdaptX/test.yml?branch=main)](https://github.com/gcuder/SoftAdaptX/actions/workflows/test.yml)


# SoftAdapt

This repository contains an updated implementation of the [SoftAdapt algorithm](https://arxiv.org/pdf/1912.12355.pdf)(techniques for adaptive loss balancing of multi-tasking neural networks). Since 2020 (when SoftAdapt was first published), SoftAdapt has been applied to a variety of applications, ranging from generative models (e.g. these papers for [VAEs](https://arxiv.org/abs/2009.11693) and [GANs](https://www.sciencedirect.com/science/article/pii/S0167739X2100488X)), [model compression](https://arxiv.org/abs/2012.01604), and [Physics Informed Neural Networks](https://arxiv.org/pdf/2211.16753.pdf), to name a few.

**SoftAdapt is also available within [NVIDIA's Modulus](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/theory/advanced_schemes.html#:~:text=annular%20ring%20example.-,SoftAdapt,-Softadapt%20is%20a). [Modulus](https://developer.nvidia.com/modulus) is a an open-source framework for building, training, and fine-tuning Physics-based DL models in Python.**

[![arXiv:10.48550/arXiv.1912.12355](http://img.shields.io/badge/arXiv-110.48550/arXiv.2206.04047-A42C25.svg)](
https://doi.org/10.48550/arXiv.1912.12355)

## Installing SoftAdapt

### Using Poetry (Recommended)
SoftAdapt now uses [Poetry](https://python-poetry.org/) for dependency management. To install SoftAdapt with Poetry:

1. First, make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Then, you can install SoftAdapt directly from GitHub:
```bash
poetry add git+https://github.com/dr-aheydari/SoftAdapt.git
```

3. Or clone the repository and install locally:
```bash
git clone https://github.com/dr-aheydari/SoftAdapt.git
cd SoftAdapt
poetry install
```

### Using pip (Alternative)
If you prefer using pip, you can still install SoftAdapt:

```bash
pip install git+https://github.com/dr-aheydari/SoftAdapt.git
```

or clone and install:

```bash
git clone https://github.com/dr-aheydari/SoftAdapt.git
cd SoftAdapt
pip install .
```

## Development

### Setting Up Development Environment

If you want to contribute to SoftAdaptX, follow these steps to set up your development environment:

1. Clone the repository:
```bash
git clone https://github.com/dr-aheydari/SoftAdapt.git
cd SoftAdapt
```

2. Install development dependencies with Poetry:
```bash
poetry install
```

### Code Quality Tools

SoftAdaptX uses pre-commit hooks to ensure code quality. These hooks run automatically before each commit to check for issues and format code according to project standards.

1. Install pre-commit hooks:
```bash
poetry run pre-commit install
```

2. The following checks will run automatically on each commit:
   - ruff: For linting and formatting Python code
   - mypy: For type checking
   - Various other checks for whitespace, file endings, etc.

3. You can also run the checks manually:
```bash
poetry run pre-commit run --all-files
```

4. To update the pre-commit hooks to the latest versions:
```bash
poetry run pre-commit autoupdate
```

### Continuous Integration

SoftAdaptX uses GitHub Actions for continuous integration to ensure code quality and compatibility:

1. **Linting Workflow**: Automatically runs pre-commit hooks on all files to check code quality.
   - Runs on push to main branch and pull requests
   - Executes all pre-commit hooks including ruff, mypy, and other code quality checks

2. **Testing Workflow**: Runs unit tests across multiple Python versions.
   - Tests on Python 3.10, 3.11, and 3.12
   - Ensures compatibility across all supported Python versions
   - Runs on push to main branch and pull requests

These workflows help maintain code quality and ensure that all changes are tested before being merged.

## General Usage and Examples

SoftAdapt consists of three variants. These variants are the "original" `SoftAdapt`, `NormalizedSoftAdapt`, and `LossWeightedSoftAdapt`. Below, we discuss the logic of SoftAdapt and provide some simple examples for calculating SoftAdapt weights.

### Example 1
SoftAdapt is designed for multi-tasking neural networks, where the loss component that is being optimized consists of `n` parts (`n` > 1). For example, consider a loss function that consists of three components:

```python
criterion = loss_component_1 + loss_component_2 + loss_component_3
```
Traditionally, these loss components are weighted the same (i.e. all having coefficients of 1); however, as shown by many studies, using different balancing coefficients for each component based on the optimization performance can significantly improve model training. SoftAdapt aims to calculate the most optimal set of (convex) weights based on live statistics.

Considering the example above, let us assume that the first 5 epochs have resulted in the following loss values:
```python
loss_component_1 = torch.tensor([1, 2, 3, 4, 5])
loss_component_2 = torch.tensor([150, 100, 50, 10, 0.1])
loss_component_3 = torch.tensor([1500, 1000, 500, 100, 1])
```
Clearly, the first loss component is not being optimized as well as the other two parts, since it is increasing while the rates of change for component 2 and 3 being negative (with component 3 decreasing 10x faster than component 2). Now let us see the different variants of SoftAdapt in action for this problem.

```python
from softadaptx import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
import torch

# We redefine the loss components above for the sake of completeness.
loss_component_1 = torch.tensor([1, 2, 3, 4, 5])
loss_component_2 = torch.tensor([150, 100, 50, 10, 0.1])
loss_component_3 = torch.tensor([1500, 1000, 500, 100, 1])

# Here we define the different SoftAdapt objects
softadapt_object = SoftAdapt(beta=0.1)
normalized_softadapt_object = NormalizedSoftAdapt(beta=0.1)
loss_weighted_softadapt_object = LossWeightedSoftAdapt(beta=0.1)
```
(1) The original variant calculations are:
```python
softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3)
# >>> tensor([9.9343e-01, 6.5666e-03, 3.8908e-22], dtype=torch.float64)
```
(2) Normalized slopes variant outputs:
```python
normalized_softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3)
# >>> tensor([0.3221, 0.3251, 0.3528], dtype=torch.float64)
```
and (3) the loss-weighted variant results in:
 ```python
loss_weighted_softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3)
#>>> tensor([8.7978e-01, 1.2022e-01, 7.1234e-20]
```
as we see above, the first variant only focuses on the rates of change of each loss function, and since the values in component 3 are decreasing much faster than the other two components, the algorithm assigns it a weight very close to 0. Similarly, the second component also gets a very small weight while the first component has a weight close to 1. This means that the optimzer should primarily focus on the first component, and in a sense, not worry about components 2 and 3. On the other hand, the second variant normalizes the slopes, which significantly reduces the differences in the rate of change, resulting in a much more moderate distribution of weights across the three components. Lastly, Loss-Weighted SoftAdapt not only considers the rates of change, but also considers the value of loss functions (an average of each over the last `n` iterations, in this case `n=5`). Though the first component still receives the highest attention value in the Loss-Weighted SoftAdapt, the value of the second component is slightly larger than in the first case.

### Example 2
Let us consider the same three loss components as before, but now with another loss that is performing the worst. That is:

```python
loss_component_1 = torch.tensor([1, 2, 3, 4, 5])
loss_component_2 = torch.tensor([150, 100, 50, 10, 0.1])
loss_component_3 = torch.tensor([1500, 1000, 500, 100, 1])
loss_component_4 = torch.tensor([10, 20, 30, 40, 50])
```
Intuitively, the fourth loss function should recieve the most amount of attention from the optimizer, followed by the component 1. The loss components 2 and 3 should recieve the least, with component 2 recieving slightly higher weight than component 3. Using the same objects we defined in Example 1, we now want to see how each variant calculates the weights:

(1) `SoftAdapt`:
```python
softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3, loss_component_4)
# >>> tensor([2.8850e-01, 1.9070e-03, 1.1299e-22, 7.0959e-01], dtype=torch.float64)
```
(2) `NormalizedSoftAdapt`:
```python
normalized_softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3, loss_component_4)
# >>> tensor([0.2436, 0.2459, 0.2673, 0.2432]
```
(3) `LossWeightedSoftAdapt`
 ```python
loss_weighted_softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3, loss_component_4)
#>>> tensor([3.8861e-02, 5.3104e-03, 3.1465e-21, 9.5583e-01], dtype=torch.float64)
```
As before, we see that `SoftAdapt` and `LossWeightedSoftAdapt` follow our intuition more closely.

***In general, we highly recommend using the loss-weighted variant of SoftAdapt since it considers a running average of previous loss values as well as the rates of change***.

## Usage for Training Neural Networks

SoftAdapt can be easily used as an add-on to your existing training scripts, with the logic being simple and straight forward. SoftAdapt requires you to keep track of each loss component seperatly, and call the appropriate SoftAdapt variant after `n` iterations (as defined by user). The following example demonstrate how these changes should take place in your existing code:

```python
# Assume `model`, `optimizer` and the dataloader are defined
# We assume optimization is being done with 3 loss function
loss_component_1 = MyCriterion1()
loss_component_2 = MyCriterion2()
loss_component_3 = MyCriterion3()

# Main training loop:
for current_epoch in range(training_epochs):
  for batch_idx, data in enumerate(train_data_loader):
      features, labels = data
      optimizer.zero_grad()
      outputs, _, _ = model(features.float())
      loss = loss_component_1(outputs) + loss_component_2(outputs) + loss_component_3(outputs)

      loss.backward()
      optimizer.step()

```

Here are the only changes that need to be added to the above code to utilize SoftAdapt in training:

```python
loss_component_1 = MyCriterion1()
loss_component_2 = MyCriterion2()
loss_component_3 = MyCriterion3()

# Change 1: Create a SoftAdapt object (with your desired variant)
softadapt_object = LossWeightedSoftAdapt(beta=0.1)

# Change 2: Define how often SoftAdapt calculate weights for the loss components
epochs_to_make_updates = 5

# Change 3: Initialize lists to keep track of loss values over the epochs we defined above
values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []
# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1,1])

# Main training loop:
for current_epoch in range(training_epochs):
  for batch_idx, data in enumerate(train_data_loader):
      features, labels = data
      optimizer.zero_grad()
      outputs, _, _ = model(features.float())
      # Keeping track of each loss component
      values_of_component_1.append(loss_component_1(outputs))
      values_of_component_2.append(loss_component_2(outputs))
      values_of_component_3.append(loss_component_3(outputs))

      # Change 4: Make sure `epochs_to_make_change` have passed before calling SoftAdapt.
      if current_epoch % epochs_to_make_updates == 0 and current_epoch != 0:
          adapt_weights = softadapt_object.get_component_weights(torch.tensor(values_of_component_1),
                                                                 torch.tensor(values_of_component_2),
                                                                 torch.tensor(values_of_component_3),
                                                                 verbose=False,
                                                                 )


          # Resetting the lists to start fresh (this part is optional)
          values_of_component_1 = []
          values_of_component_2 = []
          values_of_component_3 = []

      # Change 5: Update the loss function with the linear combination of all components.
      loss = adapt_weights[0] * loss_component_1(outputs) + adapt_weights[1]*loss_component_2(outputs) + adapt_weights[2]*loss_component_3(outputs)

      loss.backward()
      optimizer.step()

```

With this previous usage example, the weights are updated for every batch of an epoch when `current_epoch % epochs_to_make_updates == 0` condition is met. Similar to mini-batch gradient descent, the idea is that updating the adaptive weights per batch allows to learn from smaller chunks of data and potentially converge faster (at the cost of possibly introducing a bit of noise). In most applications, this approach provides the best convergence.

One could also update the weights once before the optimization loop over mini batches, like in this example where weights are updated only once every 5 epochs:

```python
loss_component_1 = MyCriterion1()
loss_component_2 = MyCriterion2()
loss_component_3 = MyCriterion3()

# Change 1: Create a SoftAdapt object (with your desired variant)
softadapt_object = LossWeightedSoftAdapt(beta=0.1)

# Change 2: Define how often SoftAdapt calculate weights for the loss components
epochs_to_make_updates = 5

# Change 3: Initialize lists to keep track of loss values over the epochs we defined above
values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []
# Initializing adaptive weights to all ones.
adapt_weights = torch.tensor([1,1,1])

# Main training loop:
for current_epoch in range(training_epochs):
    # Change 4: Make sure `epochs_to_make_change` have passed before calling SoftAdapt.
    if current_epoch % epochs_to_make_updates == 0 and current_epoch != 0:
        adapt_weights = softadapt_object.get_component_weights(torch.tensor(values_of_component_1),
                                                               torch.tensor(values_of_component_2),
                                                               torch.tensor(values_of_component_3),
                                                               verbose=False,
                                                               )

        # Resetting the lists to start fresh (this part is optional)
        values_of_component_1 = []
        values_of_component_2 = []
        values_of_component_3 = []

    for batch_idx, data in enumerate(train_data_loader):
        features, labels = data
        optimizer.zero_grad()
        outputs, _, _ = model(features.float())
        # Keeping track of each loss component
        values_of_component_1.append(loss_component_1(outputs))
        values_of_component_2.append(loss_component_2(outputs))
        values_of_component_3.append(loss_component_3(outputs))

        # Change 5: Update the loss function with the linear combination of all components.
        loss = adapt_weights[0] * loss_component_1(outputs) + adapt_weights[1]*loss_component_2(outputs) + adapt_weights[2]*loss_component_3(outputs)

        loss.backward()
        optimizer.step()

```

*Please feel free to open issues for any questions!*


## Citing Our Work
If you found our work useful for your research, please cite us as:
```
@article{DBLP:journals/corr/abs-1912-12355,
  author    = {A. Ali Heydari and
               Craig A. Thompson and
               Asif Mehmood},
  title     = {SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks
               with Multi-Part Loss Functions},
  journal   = {CoRR},
  volume    = {abs/1912.12355},
  year      = {2019},
  url       = {http://arxiv.org/abs/1912.12355},
  eprinttype = {arXiv},
  eprint    = {1912.12355},
  timestamp = {Fri, 03 Jan 2020 16:10:45 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1912-12355.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
