# Fourier123: One Image to High-Quality 3D Object Generation with Hybrid Fourier Score Distillation
[Shuzhou Yang](https://ysz2022.github.io/), [Yu Wang](https://villa.jianzhang.tech/people/yu-wang-%E6%B1%AA%E7%8E%89/), [Haijie Li](https://villa.jianzhang.tech/people/haijie-li-%E6%9D%8E%E6%B5%B7%E6%9D%B0/), [Jiarui Meng](), [Xiandong Meng](), [Jian Zhang](https://jianzhang.tech/)

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)]()
[![Home Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://fourier1-to-3.github.io/)



# Abstract
Single image-to-3D generation is pivotal for crafting controllable 3D assets. Given its underconstrained nature, we leverage geometric priors from a 3D novel view generation diffusion model and appearance priors from a 2D image generation method to guide the optimization process. We note that a disparity exists between the training datasets of 2D and 3D diffusion models, leading to their outputs showing marked differences in appearance. Specifically, 2D models tend to deliver more detailed visuals, whereas 3D models produce consistent yet over-smooth results across different views. Hence, we optimize a set of 3D Gaussians using 3D priors in spatial domain to ensure geometric consistency, while exploiting 2D priors in the frequency domain through Fourier transform for higher visual quality. This 2D-3D **hy**brid **F**ourier **S**core **D**istillation objective function (dubbed **hy-FSD**), can be integrated into existing 3D generation methods, yielding significant performance improvements. With this technique, we further develop an image-to-3D generation pipeline to create high-quality 3D objects within one minute, named **Fourier123**. Extensive experiments demonstrate that Fourier123 excels in efficient generation with rapid convergence speed and visual-friendly generation results.
