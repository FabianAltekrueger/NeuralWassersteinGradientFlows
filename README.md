# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels

This code belongs to the paper [1]. Please cite the paper, if you use this code.

The paper [1] is available at https://arxiv.org/abs/2301.11624.

The repository contains an implementation of Neural backward scheme and Neural forward scheme as introduced in [1]. It contains scripts for reproducing the numerical examples in Section 6.

For questions and bug reports, please contact Fabian Altekrueger (fabian.altekrueger@hu-berlin.de) or Johannes Hertrich (j.hertrich@math.tu-berlin.de).

## CONTENTS

1. REQUIREMENTS  
2. USAGE AND EXAMPLES
3. REFERENCES

## 1. REQUIREMENTS

The code requires several Python packages. We tested the code with Python 3.9.7 and the following package versions:

- pytorch 1.12.1
- numpy 1.23.3
- tqdm 4.64.1
- scipy 1.9.1
- pykeops 2.1
- matplotlib 3.6.1
- scikit-image 0.19.3

Usually the code is also compatible with some other versions of the corresponding Python packages.

## 2. USAGE AND EXAMPLES

You can start the experiments by starting the scripts `run_xxx.py`.


### Interaction Energy

The script `run_interaction.py` is the implementation of the Wasserstein gradient flow for the interaction energy. Details of the expermient can be found in Section 6.1 of [1].


<img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralForwardScheme_interaction.gif" width="250" /> &emsp; &emsp; &emsp;    <img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralBackwardScheme_interaction.gif" width="250" /> 

Neural Forward Scheme  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;Neural Backward Scheme

### Max und Moritz

The script `run_maxmoritz.py` is the implementation of the Wasserstein gradient flow for the maximum mean discrepancy (MMD) to a sampled version of the drawing 'Max und Moritz' by Wilhelm Busch. It is visualized in Section 6.2 of [1]. 

<img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralForwardScheme_MaxMoritz.gif" width="250"/> &emsp; &emsp;   <img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralBackwardScheme_MaxMoritz.gif" width="250" /> 

Neural Forward Scheme  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;  Neural Backward Scheme

### MNIST

The script `run_mnist.py` is the implementation of the Wasserstein gradient flow for the MMD to 100 handwritten digits of MNIST [2]. Details and the corresponding trajectories of the methods are given in Section 6.2 of [1].

![](https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralForwardScheme_MNIST.gif)   &emsp; &emsp; &emsp; ![](https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralBackwardScheme_MNIST.gif) 

Neural Forward Scheme  &emsp; &emsp; &emsp; &emsp;   Neural Backward Scheme

### Smiley

The script `run_smiley.py` is the implementation of the Wasserstein gradient flow for the MMD to a sampled version of the image 'Smiley'. Details can be found in Appendix F, Example 1 of [1]. 

<img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralForwardScheme_Smiley.gif" width="250" /> &emsp; &emsp;   <img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralBackwardScheme_smiley.gif" width="250" /> 

Neural Forward Scheme  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Neural Backward Scheme


### Dirac

The script `run_dirac.py` is the implementation of the Wasserstein gradient flow for the MMD to a sum of two Dirac measures visualized in Appendix F, Example 2 of [1]. 

<img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralForwardScheme_Dirac.gif" width="250" /> &emsp; &emsp; &emsp;    <img src="https://github.com/FabianAltekrueger/NeuralWassersteinGradientFlows/blob/main/animations/NeuralBackwardScheme_Dirac.gif" width="250" /> 

Neural Forward Scheme  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;  &emsp; Neural Backward Scheme

## 3. REFERENCES

[1] F. Alteküger, J. Hertrich and G. Steidl.  
Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels.   
ArXiv Preprint#2301.11624  
Accepted at: International Conference on Machine Learning 2023.

[2] Y. LeCun, L. Bottou, Y. Bengio and P.  Haffner.  
Gradient-based learning applied to document recognition.  
Proceedings of the IEEE, 86(11):2278–2324, 1998.
