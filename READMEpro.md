<h1> SGARNet: A Deep Artifact Removal Approach for Lensless Multi-Core Fiber Imaging </h1>

 **（Unpublished work）**
authors: Zewen Ma (mzw25@mails.tsinghua.edu.cn), Jinwen Wei (weijw24@mails.tsinghua.edu.cn), Juergen Czarke (juergen.czarske@tu-dresden.de), Jiachen Wu(wjc2022@tsinghua.edu.cn) and Liangcai Cao(clc@tsinghua.edu.cn)


Multi-core fiber (MCF) imaging is essential for minimally invasive endoscopy in medicine and industrial inspection. We present a lensless MCF imaging approach based on SGARNet (Spectral-Guided Artifact Removal Network). In this framework, a physics-informed prior is embedded through a lightweight SpectralGate module to suppress lattice-frequency artifacts in the feature domain.

<p align="center">
<img src="results/sample1GIF.gif" alt="SGARNet Result" width="200">
<img src="results/sample2GIF.gif" alt="SGARNet Result" width="200">
<img src="results/sample3GIF.gif" alt="SGARNet Result" width="200">
</p>


## Requirements

SGARNet algorithms are implemented  based on [NAFNet](https://github.com/megvii-research/NAFNet).

- Python 3.9.5, PyTorch >= 1.11.0, Cuda 11.3
- Platforms: Windows 10 / 11

## Dataset

- The training and testing datasets of SGARNet were acquired using a real MCF imaging system. The full datasets will be made publicly available in subsequent updates.
- Click [**here**](https://drive.google.com/drive/folders/1AjqJsuYoe0tawjgJ1BiKY2JN49Tzp2VI?usp=drive_link) to review a part of the datasets.

## Quick start

##### 1. Prepare the environment

- Download the necessary packages according to `requirements.txt`.

##### 2. Download the dataset

- The full dataset will be made publicly available in a future update.

##### 3. Run code
 
-  The code will be updated and released in a future version.



