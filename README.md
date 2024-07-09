# CMML-Pytorch
Official PyTorch implementation of CMML from the paper:

[CMML: Contextual Modulation Meta Learningfor Cold-Start Recommendation](https://arxiv.org/abs/2108.10511).

Note that we have updated our experimental results on scenario-specific setting and please check the appendix of our new arxiv version.

## Code
We offer the code of CMML for scenario-specific setting in cmml_scenario and user-specific setting in cmml_user. You can simply check these directories for specific instructions for training on CMML and main baseline algorithms.

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
[wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)
[lmzintgraf/cavia](https://github.com/lmzintgraf/cavia)
[hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU)

