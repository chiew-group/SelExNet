<h2 align="center"> [MRM2026] SelExNet: A Self-Supervised Physics-Informed Framework for Multi-Channel Joint RF and Gradient Waveform Optimization in 2D Spatially Selective Excitation </h2>

<p align="center">
<a href="https://onlinelibrary.wiley.com/doi/10.1002/mrm.70431"><img src="https://img.shields.io/badge/Wiley-Paper-red"></a>
<a href="http://yuliangxiao.com/SelExNet-Webpage/"><img src="https://img.shields.io/badge/Web-Page-magenta"></a>
</p>

<div align="center">
  <a href='https://scholar.google.com/citations?user=e1nU4X0AAAAJ' target='_blank'><strong>Yuliang Xiao</strong></a><sup> 1,2</sup>,&thinsp;
  <a href='https://orcid.org/0009-0003-9203-7344' target='_blank'><strong>Jason Rock</strong></a><sup> 1,2</sup>,&thinsp;
  <a href='https://scholar.google.com/citations?user=NR-5WkIAAAAJ&hl=en' target='_blank'><strong>Zhe Wu</strong></a><sup> 3</sup>,&thinsp;
  <a href='https://scholar.google.com/citations?user=--juSrQAAAAJ&hl=en' target='_blank'><strong>Jamie Near</strong></a><sup> 1,2</sup>,&thinsp;
  <a href='https://scholar.google.ca/citations?hl=en&user=o8pzV3YAAAAJ' target='_blank'><strong>Mark Chiew</strong></a><sup> 1,2</sup>,&thinsp;
  <a href='https://scholar.google.ca/citations?user=wFnB60gAAAAJ&hl=en' target='_blank'><strong>Simon J. Graham</strong></a><sup> 1,2</sup></em>
</div>

<div align='center'>
  <sup>1 </sup>Sunnybrook Research Institute&ensp; <sup>2 </sup>University of Toronto&ensp;  
  <sup>3 </sup>Siemens Healthineers Limited&ensp;
</div>

<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#installation">Installation</a> |
  <a href="#train">Train</a> |
  <a href="#finetune">Finetune</a> |
  <a href="#citation">Citation</a>
</p>

## News

**2026.05.20** - We have released the code and pretrained weights for our paper. Please check the [Installation](#installation), [Train](#train) and [Finetune](#finetune) sections for details. \
**2026.05.02** - Our paper is accepted by **Magnetic Resonance in Medicine 2026**. Work in progress.

## Abstract

SelExNet is a self-supervised framework for 2D spatially selective excitation that jointly optimizes radiofrequency (RF) pulses and gradient waveforms, with extension to multi-channel transmission MRI. It couples neural RF and gradient generators with a differentiable Bloch simulator, enabling pulse optimization directly from desired excitation outcomes without requiring pre-designed target pulses.

The framework designs RF pulses and parameterized variable-density spiral gradient waveforms for both single- (sTx) and multi-channel (pTx) transmission, and supports patient-specific adaptation using measured, previously unseen <em>B</em><sub>0</sub> and
<em>B</em><sub>1</sub><sup>+</sup> maps. Joint RF-Gradient optimization improves excitation fidelity over RF-only optimization. In phantom experiments, patient-specific fine-tuning restores target geometry and uniformity under field inhomogeneity. In-vivo studies demonstrate anatomically precise excitation, sharper target boundaries, and reduced off-target signal.

<div align="center">
  <img src="./assets/workflow.png" alt="SelExNet Workflow" width="800"><br>
  <b>Figure 1: Overview of SelExNet framework.</b>
</div>

## Installation

`conda` is recommended for creating the Python environment. After activating the environment, install `uv` and use it for faster package installation:

```bash
conda create -n selexnet python=3.10
conda activate selexnet
pip install uv
git clone https://github.com/chiew-group/SelExNet.git
cd SelExNet
```

Install SelExNet and its required dependencies from the repository root:

```bash
uv pip install -e .
```

For the optional imaging dependencies, install:

```bash
uv pip install -e ".[all]"
```

For development tools, install:

```bash
uv pip install -e ".[dev]"
```

### Dependencies

<details open>

<summary><b>Core dependencies (installed via <code>uv pip install -e .</code>)</b></summary>

- torch
- omegaconf
- tqdm
- scipy
- numpy
- pytorch-msssim
- torch-pca
- opencv-python
- matplotlib
- matplotlib-inline
- Pillow

</details>

<details>

<summary><b>Optional imaging dependencies (installed via <code>uv pip install -e ".[all]"</code>)</b></summary>

- torchvision
- pydicom
- scikit-image
- scikit-learn

</details>

<details>

<summary><b>Development dependencies (installed via <code>uv pip install -e ".[dev]"</code>)</b></summary>

- selexnet[all]
- pytest
- black
- flake8
- isort
- mypy
- jupyterlab

</details>

### Verify Installation

```bash
python -c "import selexnet; print(selexnet.__version__)"
```

## Train

> [!TIP]
> The pretraining for 8ch pTx usually takes ~2 days, we recommend using the pretrained weights for fine-tuning, see the [Finetune](#finetune) section.

### Prepare your own dataset and field maps

The dataset for this project is simple which are just binary masks with different geometries. You can either download our dataset from [Google Drive](https://drive.google.com/file/d/1LcqnGLX_euuBnytbGpVcrqFcYGSqbnys/view?usp=sharing), and uncompress and place it in the `data/train/` directory, or generate by yourself. The field maps can be in either `single-map` mode or `multi-map` mode, which is determined by the `use_fm_generator` flag in the configuration file for training. Format and structure of the field maps are described as follows:

- single-map (npz format with keys: `b0`, `b1_real`, `b1_imag`, unit is `Tesla`):
  - b0: (H, W) array of B0 map
  - b1_real: (Tx, H, W) array of real part of B1+ map
  - b1_imag: (Tx, H, W) array of imaginary part of B1+ map

- multi-map (npz format with keys: `b0_{mode}`, `b1_real_{mode}`, `b1_imag_{mode}` where the mode can be `train`, `valid` or `test`, unit is `Tesla`):
  - b0_train: `(N_train, H, W)` array of B0 maps for training
  - b0_valid: `(N_valid, H, W)` array of B0 maps for validation
  - b0_test: `(N_test, H, W)` array of B0 maps for testing
  - b1_real_train: `(N_train, Tx, H, W)` array of real part of B1+ maps for training
  - b1_imag_train: `(N_train, Tx, H, W)` array of imaginary part of B1+ maps for training
  - b1_real_valid: `(N_valid, Tx, H, W)` array of real part of B1+ maps for validation
  - b1_imag_valid: `(N_valid, Tx, H, W)` array of imaginary part of B1+ maps for validation
  - b1_real_test: `(N_test, Tx, H, W)` array of real part of B1+ maps for testing
  - b1_imag_test: `(N_test, Tx, H, W)` array of imaginary part of B1+ maps for testing

> [!TIP]
> The `single-map` mode is for training a model that is specific to the provided field maps, while the `multi-map` mode is for training a model that can adapt to unseen field maps. For the `multi-map` mode in our experiments, we found it is not easy and stable to optimize a model that can adapt to large sets of field maps. We highly recommend using a small sets or `single-map` mode for training, and then fine-tuning the model on the target field maps.

The dataset should be organized in the following structure:

```data/
├──data/
│    ├── train/
│    │   ├── fm/
│    │   │   ├── field_map.npz  # contains B0 and B1+ maps
│    │   │   └── loading_mask.npy # contains the binary mask of the loading
│    │   └── dataset/
│    │       ├── sample_1.png  # sample image 1
│    │       ├── sample_2.png  # sample image 2
│    │       └── ...           # more samples
│    ├── finetune/
```

### Train the model

```bash
python main.py \
--cfg configs/train_config.yaml \
--resume # resume the training from the last checkpoint, otherwise ignore this flag
```

If you have multiple GPUs, you can use distributed training to speed up the training process. You can launch it with the following command:

```bash
torchrun --nproc_per_node=<num_gpus> --standalone main.py \
--cfg configs/train_config.yaml \
--resume \  # resume the training from the last checkpoint, otherwise ignore this flag
--ddp # use distributed data parallel training
```

The configuration file is in `YAML` format. You can find an example configuration file at `configs/train_config.yaml`. You can also modify the parameters in the configuration file according to your needs. All the training outputs, including the trained model checkpoints and the training logs, will be saved in the `outputs/exp_name` directory.

## Finetune

After training the model, you can fine-tune it on the unseen field maps to further improve the excitation performance. We provided pretrained weights, example field maps and shape which can be used for fine-tuning. You can download them:

- [Pretrained weights](https://drive.google.com/file/d/1NygxKzWy38TgDMXBbbckgFKDoBgFsCSG/view?usp=sharing): place it in the `data/finetune/pretrained/` directory.

- [Example field maps](https://drive.google.com/file/d/18j0v21U9en84BntP2elibEFWe_HrEce1/view?usp=sharing): place it in the `data/finetune/fm/` directory.

- [Example phantom mask](https://drive.google.com/file/d/17pSShyDPUAjQVSgE6IGMLer60wP0Apln/view?usp=sharing): place it in the `data/finetune/fm/` directory.

> [!IMPORTANT]
> Make a copy of excitation shape image to the `data/finetune/dataset/` directory for fine-tuning (see the example in `data/finetune/dataset/`). This is due to the fact that the fine-tuning is a kind of overfitting process which needs same image for both training and validation.

### Finetune the model

```bash
python finetune.py \
--cfg configs/finetune_config.yaml \
--finetune_weights <path/to/finetune/pretrained.pt>
```

For demo, the `--finetune_weights` should be set to `data/finetune/pretrained/pretrained.pt`. The configuration file is in `YAML` format. You can find an example configuration file at `configs/finetune_config.yaml`. You can also modify the parameters in the configuration file according to your needs. All the fine-tuning outputs, including the fine-tuned model checkpoints and the fine-tuning logs, will be saved in the `outputs/exp_name` directory. Once the fine-tuning is done, you can use the fine-tuned model for inference on the target excitation shape and field maps.

```bash
python test_single.py \
--cfg configs/finetune_config.yaml \
--output_dir <path/to/output/dir> \
--img_path <path/to/shape/image.png>
```

In the demo, the `--output_dir` is set to `demo/` and `--img_path` should be `data/finetune/dataset/shape.png`. Finally, the simulated excitation pattern, and tailored RF and gradient waveforms will be saved in `demo/` directory.

> [!NOTE]
> The designed RF and gradient waveforms will be saved in `.mat` format which is compatible with the native Siemens pTx workflow. If you need to use on other platforms, you should manually convert them to the required format.

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{xiao2026selexnet,
  title={SelExNet: A Self-Supervised Physics-Informed Framework for Multi-Channel Joint RF and Gradient Waveform Optimization in 2D Spatially Selective Excitation},
  author={Xiao, Yuliang and Rock, Jason and Wu, Zhe and Near, Jamie and Chiew, Mark and Graham, Simon J.},
  journal={Magnetic Resonance in Medicine},
  volume = {96},
  number = {3},
  pages = {1219-1234},
  doi = {https://doi.org/10.1002/mrm.70431},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.70431},
  year={2026},
}
```
