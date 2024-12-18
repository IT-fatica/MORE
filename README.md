# MORE: Molecule Pretraining with Multi-Level Pretext Task
<p align="center">
<img src=overview.jpg width=900px>
</p>

## Installation

We used the following packages under `Python 3.7`.

```
pytorch 1.13.1
torch-cluster 1.6.1
torch-geometric 2.6.0
torch-scatter 2.1.1
torch-sparse 0.6.17
rdkit 2022.9.5
```

## Pretrain step
Please run `pretraining.py` for downstream adaptations. 

The pre-trained models we use follow the training steps of the paper [*Strategies for Pre-training Graph Neural Networks*](https://github.com/snap-stanford/pretrain-gnns) and [*GraphMAE*](https://github.com/THUDM/GraphMAE/tree/main/chem)


## Dataset
The pre-training and downstream datasets used in our experiments are referred to the paper *Strategies for Pre-training Graph Neural Networks*. You can download the biology and chemistry datasets from [their repository](https://github.com/snap-stanford/pretrain-gnns). 

- To run the codes successfully, the downloaded datasets should be placed in `/dataset_conf` and `/dataset_info` for pre-training

(If you're using 3D-level pretext task, you'll need to use the `/dataset_conf`)

(If you are not using 3D-level pretext task, you'll need to use the `/dataset_info`)

- To run the codes successfully, the downloaded datasets should be placed in `/dataset` for fine-tuning



We use `Pretrain/dataset_conf/zinc_2m_MD` and `Pretrain/dataset_info/zinc_2m_MD`

(Preprocessed data from zinc_standard_agent dataset, you can get [here](https://drive.google.com/drive/folders/1SDz7uzOk_GA17LPO-K-Lc0tGmyQlJSbK?usp=sharing))


## Fine-tune step
Please run `finetune.py` for downstream adaptations. 

We provide pretrained MORE (Finetune/pretrain/MORE.pth)


## Example
For pretraining, `Pretrain/example.ipynb`

For Fine-tuning, `Finetune/example.ipynb`

