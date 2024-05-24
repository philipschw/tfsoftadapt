# SoftAdapt

This repository contains an implementation of the [SoftAdapt algorithm](https://arxiv.org/pdf/1912.12355.pdf)(techniques for adaptive loss balancing of multi-tasking neural networks) for TensorFlow. We made essentially a one-to-one copy of [SoftAdapt](https://github.com/dr-aheydari/SoftAdapt) and replaced `torch` functions by `tensorflow` functions. All credit and glory belongs to the quoted authors and repository. Thank you very much!

[![arXiv:10.48550/arXiv.1912.12355](http://img.shields.io/badge/arXiv-110.48550/arXiv.2206.04047-A42C25.svg)](
https://doi.org/10.48550/arXiv.1912.12355)

## Installing TFSoftAdapt
### Installing the GitHub Repository (Recommended)
TFSoftAdapt can be cloned and then installed as the following:
```
$ git clone https://github.com/philipschw/tfsoftadapt.git
$ pip install ./tfsoftadapt
```

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
from tfsoftadapt import TFSoftAdapt, TFNormalizedSoftAdapt, TFLossWeightedSoftAdapt
import tensorflow as tf
# We redefine the loss components above for the sake of completeness.
loss_component_1 = tf.constant([1, 2, 3, 4, 5])
loss_component_2 = tf.constant([150, 100, 50, 10, 0.1])
loss_component_3 = tf.constant([1500, 1000, 500, 100, 1])

# Here we define the different SoftAdapt objects
softadapt_object  = TFSoftAdapt(beta=0.1)
normalized_softadapt_object  = TFNormalizedSoftAdapt(beta=0.1)
loss_weighted_softadapt_object  = TFLossWeightedSoftAdapt(beta=0.1)
```
(1) The original variant calculations are: 
```python
softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([9.9343336e-01, 6.5666283e-03, 3.8908041e-22], dtype=float32)>
```
(2) Normalized slopes variant outputs:
```python
normalized_softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.32207233, 0.32507923, 0.35284847], dtype=float32)>
```
and (3) the loss-weighted variant results in:
 ```python
loss_weighted_softadapt_object.get_component_weights(loss_component_1, loss_component_2, loss_component_3)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([8.79777193e-01, 1.20222814e-01, 7.12104538e-20], dtype=float32)>
```
as we see above, the first variant only focuses on the rates of change of each loss function, and since the values in component 3 are decreasing much faster than the other two components, the algorithm assigns it a weight very close to 0. Similarly, the second component also gets a very small weight while the first component has a weight close to 1. This means that the optimzer should primarily focus on the first component, and in a sense, not worry about components 2 and 3. On the other hand, the second variant normalizes the slopes, which significantly reduces the differences in the rate of change, resulting in a much more moderate distribution of weights across the three components. Lastly, Loss-Weighted SoftAdapt not only considers the rates of change, but also considers the value of loss functions (an average of each over the last `n` iterations, in this case `n=5`). Though the first component still recieves the highest attention value in the Loss-Weighted SoftAdapt, the value of the second component is slightly larger than in the first case.

## Citing their (not mine!) work.
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


