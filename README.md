# Neural-AI Final Project

_Yuxuan Zhang_ @ _[University of Florida](https://ufl.edu)_

## Objectives

1. 

## The Kay Natural Image Dataset

### labels

_labels_ is a 4 by N array of class names:

- rows `0-2` correspond to different levels of the WordNet hierarchy for the DNN predictions

- row `3` has the labels predicted by a deep neural network (DNN) trained on ImageNet

### dat

+ **Training Set** ($N = 1750$ samples)

	- `stimuli`: $N \times (W \times H)$ array of grayscale stimulus images

	- `responses`: $N \times num\_voxel$ array of z-scored BOLD response amplitude

+ **Test Set** ($M = 120$ samples)

	- `stimuli_test`: $M \times (W \times H)$ array of grayscale stimulus images in the test set.

	- `responses_test`:  $M \times num\_voxel$ array of z-scored BOLD response amplitude in the test set.

+ **Common Data**

	- $num\_voxel = 8428$. A voxel is the smallest distinguishable space unit the fMRI device could produce. It can be treated as a "volume pixel".
	These voxels are concatenated by the results from different test subjects and/or different test sessions

	- Each visual stimuli is a $128 \times 128$ grayscale image collected from online image dataset.

	- `roi`: array of voxel labels, each voxel was classified into a label according to its responses to different visual stimuli.

	- `roi_names`: names corresponding to voxel label numbers.

## The Replicated Model

According to the original research paper that came along with the dataset ([paper link](https://example.com)), the authors created a separate encoder and decoder which will be trained separately. The encoder predicts brain activity according to the visual stimuli, and the decoder predicts visual stimuli form recorded brain activity.

## Our Model



## Analysis Utilities

Many utilities were developed along with the model in this project. Their names and functionalities are listed below.

1. **Image Processors** - `util/preprocessor.py`

1. **Score Calculator** - `util/score.py`

1. **Loss Functions** - `util/loss.py`

## Citations and References

## Acknowledgements

