# T-JEPA (work in progress)

This repository aims to provide a simple implementation, inspired by Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) repo, of the Joint-Embedding Predictive Architecture (JEPA) in PyTorch for text data. The project takes inspiration from I-JEPA and V-JEPA, with some differences. All targets for a single masked sample are packed into the same input of the predictor. The predictor context inputs do not attend to the target embeddings, while the target embeddings attend to the context and their respective target chunks. The target masking strategy is taken from the UL2 paper. With the provided configs the model is trainable (no mode collapse) using the TinyStories dataset. Will need to add testing and evaluation scripts.

## Setup (Local)

### Clone the repository

Since the repository has a submodule, you need to clone it with the `--recurse-submodules` flag
```bash
git clone --recurse-submodules
```

### Create virtual environment (if you want to)
install python 3.11 with pyenv

```bash
pyenv install 3.11
```

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

### Install 

Install the required packages

```bash
pip install -r requirements.txt
```

Note: if you are running on a cloud instance you'll have to install the pytorch library with the wheel that matches the instance's CUDA version. You can find the wheel [here](https://pytorch.org/get-started/locally/)

Install the rotary embeddings package (which supports custom indexing)

```bash
pip install -e ./rotary-embedding-torch
```

Note if you don't see the rotary-embedding-torch directory, you probably forgot to clone the repository with the `--recurse-submodules` flag. You can clone the submodule again with the following command:

```bash
git submodule update --init --recursive
```

## Training

```bash
python train.py
```

you can ever modify the input arguments to the train.py script to match or add them to a set of config files and pass the files as arguments to the train.py script (see files `configs/tiny_Stories_training`).

### train.py input arguments

- `--init_from` (str, default: 'scratch'): init from `scratch` or `resume`
- `--encoder_config_path` (str, required): path to the encoder config (if init_from == 'resume' is loaded from the training directory)
- `--predictor_config_path` (str, required): path to the predictor config
- `--opt_config_path` (str, required): path to the optimizer config
- `--train_run_config_path` (str, required): path to the train run config
- `--target_masking_strategies_path` (str, required): path to the target masking strategies

  

# Target & Context masking strategy parameters

- `target_block_size` (int, required): mean of the target block size
- `max_target_mask_ratio` (float, default: 0.15): maximum ratio of the target to be masked
- `max_target_mask_start_ratio` (float, default: 0.15): maximum ratio of the target start to be masked

## Datasets

### TinyStories

run the prepare.py script to download and prepare the dataset in `data/TinyStories`

```bash
python data/TinyStories/prepare.py
```

### SlimPajama
TODO: add the prepare.py script to download and prepare the dataset in `data/SlimPajama`

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

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael, and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv preprint},
  year={2024}
}
```

```bibtex
@misc{tay2023ul2,
      title={UL2: Unifying Language Learning Paradigms}, 
      author={Yi Tay and Mostafa Dehghani and Vinh Q. Tran and Xavier Garcia and Jason Wei and Xuezhi Wang and Hyung Won Chung and Siamak Shakeri and Dara Bahri and Tal Schuster and Huaixiu Steven Zheng and Denny Zhou and Neil Houlsby and Donald Metzler},
      year={2023},
      eprint={2205.05131},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```