
[![Python 3.9.16](https://img.shields.io/badge/python-3.9.16-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fclissa%2Ffluocells-BVPAI&countColor=%23263759)

# fluocells-BVPAI
Repository with code for paper accepted to ICIAP 2023 workshop [_Beyon Vision: Physics meets AI_](https://physicsmeetsai.github.io/beyond-vision/).

## Quick Start

The repository is organized as follows:

```
fluocells-BVPAI/
├── dataset: data folder with FNC v2 yellow dataset (decompressed archive)
├── figures: paper figures and illustrations
├── fluocells: utils package for data processing, visualization, modelling and evaluation
├── models: folder with pre-trained weights
├── notebooks: example notebooks to perform data exploration, training pipeline setup and sample experiment
├── installation.txt: list of bash commands run to create paper environment
├── training.py: script to perform sample training
├── evaluate.py: script to evaluate model starting from pre-trained weights. This performs the association of true and predicted objects according to IoU overlapping and centers distance
└── compute_metrics.py: script to compute segmentation, detection and counting metrics for a list of experiments
```

### Installation

In order to install you can clone this repository and simply follow the instructions contained in [`installation.txt`](installation.txt).


## Usage

The repository contains the prototype **[fluocells](fluocells/) python package**, that collects main utils adopted to perform data manipolation, modelling and evaluation.

```
fluocells/
├── __init__.py
├── config.py: set all paths (data, metadata, models)
├── models: utils to implement and handle c-ResUnet with torch/fastai
│   ├── __init__.py
│   ├── _blocks.py
│   ├── _models.py
│   └── _utils.py
└── utils
    ├── __init__.py
    ├── annotations.py: utils to handle data annotations
    ├── data.py: utils to handle data processing
    └── metrics.py: utils to assess model performance
```

### Training
To run a sample training experiment simply run the script [`training.py`](training.py):

```
python training.py <dataset_name> [--loss loss_name] [--w_cell N] [--seed N] [--gpu_id 0]
```
- *seed* will be set randomly if not specified
- *loss* allows specifying the loss function to use for training. Options: ["Dice" (default), "BCE", "FT", "Combined"]
- Note: CombinedLoss is a weighted sum of BCE, Dice and FT terms. The corresponding weights are hard-coded in [`training.py`](training.py) script. Please refer to the code for more details


This assumes the `DATA_PATH` variable in [`fluocells/config.py`](fluocells/config.py) is correctly set to the folder where FNC `yellow/` data are, with the following structure:

```
yellow
├── test
│   ├── ground_truths
│   ├── images
│   └── metadata
├── trainval
│   ├── ground_truths
│   ├── images
│   └── metadata
└── unlabelled
    ├── images
    └── metadata

where:
./ground_truths/
├── masks: png images with binary masks
├── rle: pickle files with Running Length Encoding (RLE) of binary masks
├── Pascal_VOC: *.xml* files with image annotations (polygon, bbox, dot)
├── COCO
    └── annotations_green_trainval.json
└── VIA
    └── annotations_green_trainval.json
```

### Evaluation
In order to evaluate pre-trained models you have to run first [`evaluate.py`](evaluate.py):

```
python evaluate exp_name [--bin_thresh 0.5] [--smooth_disk 0] [--max_hole 50] [--min_size 200] [--max_dist 30] [--fp 40] [--iou_thresh 0.5] [--prox_thresh 40] [--device cpu]
```

This script loads pre-trained weights in `models/<exp_name>/model.pth`.
Performance are assesed on the test set (dataset is retrieved by the experiment name). 
The prediction include also post-processing operations like: *contours smoothing, hole/small objects removal and waterhsed*. 

The association of true and predicted objects (namely, True Positives, False Positives and False Negatives) is computed based on both i) overlapping (IoU) and ii) proximity (centers distance).
The arguments `iou_thresh` and `prox_thresh` determine the cutoff for a positive match (TP) for i) and ii), respectively.
For more details, please refer to [evaluate.py](evaluate.py) parser help.

The results are stored under the `logs/<exp_name>` folder.

After that, simply run the [`compute_metrics.py`](compute_metrics.py) script to compute *segmentation, detection* (F1 score, precision, recall) and *counting* (MAE, MedAE, MPE) metrics:

```
python compute_metrics.py exp_name1 [exp_name2 ...]
```

## Reproducibility

The results presented in the paper are obtained with the following setting.

### Training

- dataloader: please check `hyperparameter_defaults` in [training.py](training.py) 
- loss:
    - BCE, `w_cell`: [50, 100, 200]
    - Dice, `smooth`: 1e-06
    - Focal, `gamma`: 2.0
    - CombinedLoss weights (`w_bce`, `w_dice`, `w_focal`): 
        - [0.3, 0.3, 0.4] # emphasis on class unbalance
        - [0.2, 0.5, 0.3] # emphasis on segmentation (for overcrowding)
        - [0.5, 0.2, 0.5] # CellViT configuration
- `seed`: [0, 5, 25, 47, 79]

### Evaluation

In the evaluation phase, some hyperparameters affect the results. The folllowing configuration was used.

- post-processing
    - binarization threshold: 0.875
    - smooth_disk: 3
    - max_hole: 50
    - min_size: 200 # roughly equivalent to minimum size of labelled objects (expressed in pixels)
    - max_dist: 30
    - fp: 40

- evaluation
    - IoU threshold: 0.4
    - proximity threshold: 40 pixels (corresponding to X $\mu m$, average cell diameter)

