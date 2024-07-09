# CMML-Pytorch
Official PyTorch implementation of CMML from the paper:

[CMML: Contextual Modulation Meta Learningfor Cold-Start Recommendation](https://arxiv.org/abs/2108.10511).


## Platform
- python: 3.6+
- Pytorch: 1.7

## How to run CMML
```python
python3 main.py --model mlp --mlp_hyper_hidden_dim 256 --context_encoder mean
```

## How to run MELU/MetaDNN baseline
In MELU_pytorch directory run
```python
python3 maml.py --baseline MetaDNN
```
```python
python3 maml.py --baseline MELU
```
# Acknowledgement
This code refers code from:
[wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
[lmzintgraf/cavia](https://github.com/lmzintgraf/cavia).
[hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU).

