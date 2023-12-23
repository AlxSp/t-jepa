# T-JEPA

## Setup (Local)

Create a pyenv virtual environment with python 3.11

```bash
pyenv virtualenv 3.11 tjepa
```

Activate the virtual environment

```bash
pyenv activate tjepa
```

Optional (but recommended): Set the virtual environment to be automatically activated when entering the project directory

```bash
pyenv local tjepa
```

Install the required packages

```bash
pip install -r requirements.txt
```

Install the rotary embeddings package

```bash
pip install -e ./rotary-embedding-torch
```

# TODOs

## Short term

### Immediate TODOs
- [X] Device
- [X] Save/Load models
- [X] Save/Load optimizer & schedulers
- [X] Ensure reproducibility
- [ ] Log training info
- [ ] Training runs
- [ ] Target Packing
- [ ] UL2 targets
- [ ] Context/Input Packing

## Datasets

# TinyStories

max token length with LLaMa tokenizer: 1124

## Citations

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023},
  github={https://github.com/facebookresearch/ijepa}
}
```