# T-JEPA

## Setup (Local)

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

Install the required packages

```bash
pip install -r requirements.txt
```

Install the rotary embeddings package

```bash
git clone https://github.com/AlxSp/rotary-embedding-torch
cd rotary-embedding-torch/
git checkout indexing
cd ..
pip install -e ./rotary-embedding-torch
```

# TODOs

## Short term

### Immediate TODOs
- [X] Device
- [X] Save/Load models
- [X] Save/Load optimizer & schedulers
- [X] Ensure reproducibility
- [X] Log training info
- [X] Training runs
- [X] Target Packing
- [X] Mask out targets with reoccurring token sequences
- [X] UL2 targets
- [ ] Proper data prep script
- [ ] Double check target context creation
- [ ] Context/Input Packing

# Target & Context masking strategies

```python
# R denoising
{
    "target_block_size_mean" : 3,
    # "target_block_size_std" : 0.15,
    "max_target_mask_ratio" : 0.25,
    "target_block_num" : None,
},
{
    "target_block_size_mean" : 8,
    # "target_block_size_std" : 0.15,
    "max_target_mask_ratio" : 0.25,
    "target_block_num" : None,
},
# X denoising
{
    "target_block_size_mean" : 3,
    # "target_block_size_std" : 0.5,
    "max_target_mask_ratio" : 0.5,
    "target_block_num" : None,
},
{
    "target_block_size_mean" : 8,
    # "target_block_size_std" : 0.5,
    "max_target_mask_ratio" : 0.5,
    "target_block_num" : None,
},
{
    "target_block_size_mean" : 64,
    # "target_block_size_std" : 0.5,
    "max_target_mask_ratio" : 0.15,
},
{
    "target_block_size_mean" : 64,
    # "target_block_size_std" : 0.5,
    "max_target_mask_ratio" : 0.5,
},
# S denoising
{
    "target_block_size_mean" : None,
    # "target_block_size_std" : 0.5,
    "max_target_mask_start_ratio" : 0.75,
    "max_target_mask_ratio" : 0.25,
    "target_block_num" : 1,
}
```

## Datasets

# TinyStories

max token length with LLaMa tokenizer: 1124

```python
import datasets
dataset = load_dataset("roneneldan/TinyStories")
dataset.save_to_disk("data/TinyStories")
```

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
@misc{tay2023ul2,
      title={UL2: Unifying Language Learning Paradigms}, 
      author={Yi Tay and Mostafa Dehghani and Vinh Q. Tran and Xavier Garcia and Jason Wei and Xuezhi Wang and Hyung Won Chung and Siamak Shakeri and Dara Bahri and Tal Schuster and Huaixiu Steven Zheng and Denny Zhou and Neil Houlsby and Donald Metzler},
      year={2023},
      eprint={2205.05131},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```