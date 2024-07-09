# CMML-Pytorch
Official PyTorch implementation of CMML from the paper:

[CMML: Contextual Modulation Meta Learningfor Cold-Start Recommendation](https://arxiv.org/abs/2108.10511).

Note that we have updated our experimental results on scenario-specific setting and please check the appendix of our new arxiv version.

## Platform
- python: 3.6+
- Pytorch: 1.7

### Dataset

You can download the dataset used for experiments following the instructions in [ScenarioMeta](https://github.com/THUDM/ScenarioMeta)

### Training

For training on single/multi-mode dataset for s2Meta, please run `python3 src/main.py`with necessary parameters in the ScenarioMeta/src directory. All hyperparameters we use are shown in configs. 

Code of Running on multi-dataset

```python
python3 main_multimode.py --config='../configs/config-ali-movielens.json' --root_directory='../script/scenario_data/' --comment=baseline-seed-0 --dataset=ali-movielens
```

Code of Running on single-dataset(ali)

```python
python3 main.py --config='../configs/config-ali.json' --root_directory='../script/scenario_data/ali' --comment=baseline-seed-0
```

For training on single/multi-mode dataset for cmml, please run `python3 src/main.py`with necessary parameters in the CMML_scenario/src directory. All hyperparameters we use are shown in configs. 

Code of Running on multi-dataset

```python
python3 main_cmml_multimode.py --config='../configs/config-movielen-ali-hybrid-softm.json' --root_directory='../script/scenario_data/' --comment=multi-cmml-seed-0 --dataset=ali-movielens
```

Code of Running on single-dataset(ali)

```python
python3 main_cmml.py --config='../configs/config-ali-softm.json'  --root_directory='../script/scenario_data/ali' --comment=cmml-seed-0
```  

Different configurations for datasets in the paper are stored under the `configs/` directory. Launch a experiment with `--config` to specify the configuration file, `--root_directory` to specify the path to the preprocessed data, `--comment` to specify the experiment name which will be used in logging and `--gpu` to speficy the gpu id to use. 

## Cite
Please cite our paper if you use the code or datasets in your own work:
```
@article{feng2021cmml,
  title={CMML: Contextual Modulation Meta Learning for Cold-Start Recommendation},
  author={Feng, Xidong and Chen, Chen and Li, Dong and Zhao, Mengchen and Hao, Jianye and Wang, Jun},
  journal={arXiv preprint arXiv:2108.10511},
  year={2021}
}
```
# Acknowledgement
This code refers code from:
[ScenarioMeta](https://github.com/THUDM/ScenarioMeta)

