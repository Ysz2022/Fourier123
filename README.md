<div align="center">
<h2>Hybrid Fourier Score Distillation for Efficient One Image to 3D Object Generation</h2>
  
[Shuzhou Yang](https://ysz2022.github.io/), [Yu Wang](https://villa.jianzhang.tech/people/yu-wang-%E6%B1%AA%E7%8E%89/), [Haijie Li](https://villa.jianzhang.tech/people/haijie-li-%E6%9D%8E%E6%B5%B7%E6%9D%B0/), [Jiarui Meng](https://scholar.google.com/citations?user=N_pRAVAAAAAJ&hl=zh-CN), [Yanmin Wu](https://scholar.google.com/citations?user=11sQNWwAAAAJ&hl=zh-CN&oi=ao), [Xiandong Meng](), [Jian Zhang](https://jianzhang.tech/)*.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2405.20669)
[![Paper](http://img.shields.io/badge/Paper-Springer-FF6B6B.svg)](https://link.springer.com/article/10.1007/s44267-025-00089-8)
[![Project Page](https://img.shields.io/badge/Project_Page-Website-blue.svg)](https://fourier1-to-3.github.io/)

**TL;DR: Using both 2D and 3D diffusion models to generate 3D asset from a single image with hybrid fourier score distillation.**

</div>


## 🔑 Install

```bash
# Tested on: Ubuntu 20.04 with torch 2.1 & CUDA 11.8 on single RTX 3090 & 4090.
conda create --name fourier123 python=3.10
conda activate fourier123

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit
```


## 🤖 Pretrained LGM

Pretrained weight can be downloaded from [huggingface](https://huggingface.co/ashawkey/LGM).

For example, to download the fp16 model for inference:
```bash
mkdir pretrained && cd pretrained
wget -c https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..
```

For [MVDream](https://github.com/bytedance/MVDream), we use a [diffusers implementation](https://github.com/ashawkey/mvdream_diffusers).
Weights will be downloaded automatically.


## 🚀 Usage

```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py data/name.jpg

# save at a larger resolution
python process.py data/name.jpg --size 512

# process all jpg images under a dir
python process.py data

### training gaussian stage
# LGM initialization
python infer_lgm.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace <workspace> --test_path <input_image>

# Fourier123 finetuning
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/image.yaml input=<input_image> save_path=<output_name> load=<workspace>/<initialized_ply>

### 3D Gaussian visualization
CUDA_VISIBLE_DEVICES=0 python see.py --config configs/image.yaml workspace=<workspace> load=logs/<output_name>_model.ply

### Extract glb mesh from ply
python convert.py big --test_path <path to .ply file>
```

Please check `./configs/image.yaml` for more options.


### Running Example
```bash
python infer_lgm.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace workspace_test/backpack --test_path data_test/backpack_rgba.png

CUDA_VISIBLE_DEVICES=0 python main.py --config configs/image.yaml input=data_test/backpack_rgba.png save_path=backpack load=workspace_test/backpack/backpack_rgba.ply

CUDA_VISIBLE_DEVICES=0 python see.py --config configs/image.yaml workspace=workspace_test/backpack load=logs/backpack_model.ply
```

## 🤗 Tips to get better results
1. Due to the distribution of the training data for LGM, Fourier123 is sensitive to the facing direction of input images. Orthographic front-facing images always lead to good reconstructions.
2. If you get unsatisfactory results, regenerating again may have a good effect


## 🤝 Acknowledgement
We have intensively borrowed code from the following repositories. Many thanks to the authors for sharing their code.
- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [LGM](https://github.com/3DTopia/LGM)


## 📌 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{f123,
	title        = {Hybrid Fourier score distillation for efficient one image to 3D object generation},
	author       = {Shuzhou Yang and Yu Wang and Haijie Li and Jiarui Meng and Yanmin Wu and Xiandong Meng and Jian Zhang},
	year         = {2025},
	journal      = {Visual Intelligence (VI)}
}
```
