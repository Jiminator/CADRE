# Reproduction of CADRE: Contextual Attention-based Drug REsponse

## Class Paper
We published our experience of reproducing and extending CADRE in this [paper](docs/paper.pdf).

## Introduction

CADRE is an algorithm that aims for inferring the drug sensitivity of cell lines. It is robust to biological noise, has improved performance than classical machine learning models, and has better interpretability than normal deep neural networks.

We provide a way to run both the model developed by the original authors and a improved version of CADRE that uses focal loss, cosine annealing, residuals. We also turned off dropout and ReLU in the model as we found it to be suboptimal in training.

## Setup
### Config/Hardware
The experiments were carried out on A40 GPUs via NCSA Delta. The CUDA version was 11.8, so set up instructions may vary slightly based on CUDA version. The OS was Linux, and Python version was 3.11.

1. Clone the repo
```
git clone https://github.com/Jiminator/CADRE.git
cd CADRE/
```

2. Setup Conda evironement
```
conda create --name cadre python=3.11 -y
conda activate cadre
pip3 install -r requirements.txt
# Install Pytorch based on your CUDA Version/Device: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # For CUDA 11.8 
```
## Usage
### Original Model
#### Evaluation
To learn more about the passable arguments into our script, run:
```
python run_cf.py --help
```

#### Training
Use the following script for training the original CADRE model on the GDSC dataset:

```
SEEDS=(5497 58475 94707)
export PYTHONHASHSEED=${SEEDS[0]}
python run_cf.py --repository gdsc --model_label cntx-attn-gdsc --seed=${SEEDS[0]} --store_model=True --pkl_path=original.pkl
```

#### Evaluation
Use the following script for evaluating the original CADRE model on the GDSC dataset:

```
SEEDS=(5497 58475 94707)
export PYTHONHASHSEED=${SEEDS[0]}
python run_cf.py --repository gdsc --model_label cntx-attn-gdsc --seed=${SEEDS[0]} --eval=True --pkl_path=original.pkl
```

### Our Model
#### Training
To train our version of CADRE, run this following below
```
SEEDS=(5497 58475 94707)
export PYTHONHASHSEED=${SEEDS[0]}
python run_cf.py --repository gdsc --model_label cntx-attn-gdsc --seed=${SEEDS[0]} --dropout_rate=0.0 --use_relu=False --focal=True --alpha=0.7 --scheduler=cosine --use_residual=True --store_model=True --pkl_path=new.pkl
```

#### Evaluation
Use the following script for evaluation our CADRE model on the GDSC dataset:

```
SEEDS=(5497 58475 94707)
export PYTHONHASHSEED=${SEEDS[0]}
python run_cf.py --repository gdsc --model_label cntx-attn-gdsc --seed=${SEEDS[0]} --dropout_rate=0.0 --use_relu=False --focal=True --alpha=0.7 --scheduler=cosine --use_residual=True --eval=True --pkl_path=new.pkl
```

The output will be saved to `data/output/cf/`.

## Data

The authors prepared the preprocessed GDSC data under the directory `data/input/`. 
* `gdsc.csv`: Discretized binary responses of cell lines to drugs.
* `exp_gdsc.csv`: Discretized binary gene expression levels of cell lines.
* `mut_gdsc.csv`: Discretized binary gene mutations of cell lines (Not used in the final work due to lack of information).
* `cnv_gdsc.csv`: Discretized binary gene CNVs of cell lines (Not used in the final work due to lack of information).
* `met_gdsc.csv`: Discretized binary gene methylations of cell lines (Not used in the final work due to lack of information).
* `drug_info_gdsc.csv`: Information of drugs in the GDSC dataset.
* `exp_emb_gdsc.csv`: 200-dim gene embeddings extracted from the Gene2Vec algorithm.
* `rng.txt`: Random number generator for splitting the dataset.

To select the appropriate dataset, just change the `--exp` option when running the above commands.

## Acknowledgement

This repository was heavily borrowed from the original author's codebase. If you find this repository helpful, please cite the original author's paper: 
Yifeng Tao<sup>＊</sup>, Shuangxia Ren<sup>＊</sup>, Michael Q. Ding, Russell Schwartz<sup>†</sup>, Xinghua Lu<sup>†</sup>. [**Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention**](http://proceedings.mlr.press/v126/tao20a.html). Proceedings of the Machine Learning for Healthcare Conference (***MLHC***). 2020.
```
@inproceedings{tao2020cadre,
  title = {Predicting Drug Sensitivity of Cancer Cell Lines via Collaborative Filtering with Contextual Attention},
  author = {Tao, Yifeng  and  Ren, Shuangxia  and  Ding, Michael Q.  and  Schwartz, Russell  and  Lu, Xinghua},
  series = {Proceedings of Machine Learning Research},
  volume = {126},
  pages = {660--684},
  year = {2020},
  month = {07--08 Aug},
  editor = {Finale Doshi-Velez and Jim Fackler and Ken Jung and David Kale and Rajesh Ranganath and Byron Wallace and Jenna Wiens},
  address = {Virtual},
  publisher = {PMLR},
  url = {http://proceedings.mlr.press/v126/tao20a.html},
  pdf = {http://proceedings.mlr.press/v126/tao20a/tao20a.pdf},
}
```
