<div align="center">

# <b>GPS-Gaussian</b>: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis

[Shunyuan Zheng](https://shunyuanzheng.github.io)<sup>&dagger;,1</sup>, [Boyao Zhou](https://yaourtb.github.io)<sup>2</sup>, [Ruizhi Shao](https://dsaurus.github.io/saurus)<sup>2</sup>, [Boning Liu](https://liuboning2.github.io)<sup>2</sup>, [Shengping Zhang](http://homepage.hit.edu.cn/zhangshengping)<sup>*,1,3</sup>, [Liqiang Nie](https://liqiangnie.github.io)<sup>1</sup>, [Yebin Liu](https://www.liuyebin.com)<sup>2</sup>

<p><sup>1</sup>Harbin Institute of Technology &nbsp;&nbsp;<sup>2</sup>Tsinghua Univserity &nbsp;&nbsp;<sup>3</sup>Peng Cheng Laboratory
<br><sup>*</sup>Corresponding author &nbsp;&nbsp;<sup>&dagger;</sup>Work done during an internship at Tsinghua Univserity<p>

### [Projectpage](https://shunyuanzheng.github.io/GPS-Gaussian) · [Video](https://youtu.be/HjnBAqjGIAo) · [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_GPS-Gaussian_Generalizable_Pixel-wise_3D_Gaussian_Splatting_for_Real-time_Human_Novel_CVPR_2024_paper.pdf) · [Supp.](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zheng_GPS-Gaussian_Generalizable_Pixel-wise_CVPR_2024_supplemental.pdf)

</div>

## Introduction

We propose GPS-Gaussian, a generalizable pixel-wise 3D Gaussian representation for synthesizing novel views of any unseen characters instantly without any fine-tuning or optimization.

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/54a253ad-012a-448f-8303-168d80d3f594

## Installation

To deploy and run GPS-Gaussian, run the following scripts:
```
conda env create --file environment.yml
conda activate gps_gaussian
```
Then, compile the ```diff-gaussian-rasterization``` in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository:
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..
```
(optinal) [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) provides a faster CUDA implementation of the correlation sampler to speed up the model without impacting performance:
```
git clone https://github.com/princeton-vl/RAFT-Stereo.git
cd RAFT-Stereo/sampler && python setup.py install && cd ../..
```
If compiled this CUDA implementation, set ```corr_implementation='reg_cuda'``` in [config/stereo_human_config.py](config/stereo_human_config.py#L33) else ```corr_implementation='reg'```.

## Run on synthetic human dataset

### Dataset Preparation
- We provide rendered THuman2.0 dataset for GPS-Gaussian training in 16-camera setting, download ```render_data``` from [Baidu Netdisk](https://pan.baidu.com/s/1sX9m8wRDSQAI9d78wST7mw?pwd=rax4) or [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/sawyer0503_hit_edu_cn/EkE2GFd2saBCh_XkY3TsoV0BVTmK1UiTTKJDYje3U3vdkw?e=YazWdd) and unzip it. Since we recommend rectifying the source images and determining the disparity in an offline manner, the saved files and the downloaded data necessity around 50GB of free storage space.
- To train a more robust model, we recommend collecting more human scans for training (<em>e.g.</em> [Twindom](https://web.twindom.com), [Render People](https://renderpeople.com/), [2K2K](https://sanghunhan92.github.io/conference/2K2K/)). Then, render the training data as the target scenario, including the number of cameras and the radius of the scene. We provide the rendering code to generate training data from human scans, see [data documentation](prepare_data/MAKE_DATA.md) for more details.

### Training
Note: At the first training time, we do stereo rectify and determine the disparity offline, the processed data will be saved at ```render_data/rectified_local```. This process takes several hours and can extremely speed up the following training scheme. If you want to skip this pre-processing, set ```use_processed_data=False``` in [stage1.yaml](config/stage1.yaml#L11) and [stage2.yaml](config/stage2.yaml#L15).

- Stage1: pretrain the depth prediction model. Set ```data_root``` in [stage1.yaml](config/stage1.yaml#L12) to the path of unzipped folder ```render_data```.
```
python train_stage1.py
```

- Stage2: train the full model. Set ```data_root``` in [stage2.yaml](config/stage2.yaml#L16) to the path of unzipped folder ```render_data```, and set the correct pretrained stage1 model path ```stage1_ckpt``` in [stage2.yaml](config/stage2.yaml#L3)
```
python train_stage2.py
```
- We provide the pretrained model ```GPS-GS_stage2_final.pth``` in [Baidu Netdisk](https://pan.baidu.com/s/1sX9m8wRDSQAI9d78wST7mw?pwd=rax4) and [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/sawyer0503_hit_edu_cn/EkE2GFd2saBCh_XkY3TsoV0BVTmK1UiTTKJDYje3U3vdkw?e=YazWdd) for fast evaluation and testing.

### Testing

- Real-world data: download the test data ```real_data``` from [Baidu Netdisk](https://pan.baidu.com/s/1sX9m8wRDSQAI9d78wST7mw?pwd=rax4) or [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/sawyer0503_hit_edu_cn/EkE2GFd2saBCh_XkY3TsoV0BVTmK1UiTTKJDYje3U3vdkw?e=YazWdd). Then, run the following code for synthesizing a fixed novel view between ```src_view``` 0 and 1, the position of novel viewpoint between source views is adjusted with a ```ratio``` ranging from 0 to 1. 
```
python test_real_data.py \
--test_data_root 'PATH/TO/REAL_DATA' \
--ckpt_path 'PATH/TO/GPS-GS_stage2_final.pth' \
--src_view 0 1 \
--ratio=0.5
```

- Freeview rendering: run the following code to interpolate freeview between source views, and modify the ```novel_view_nums``` to set a specific number of novel viewpoints.
```
python test_view_interp.py \
--test_data_root 'PATH/TO/RENDER_DATA/val' \
--ckpt_path 'PATH/TO/GPS-GS_stage2_final.pth' \
--novel_view_nums 5
```

# Citation

If you find this code useful for your research, please consider citing:
```bibtex
@inproceedings{zheng2024gpsgaussian,
  title={GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis},
  author={Zheng, Shunyuan and Zhou, Boyao and Shao, Ruizhi and Liu, Boning and Zhang, Shengping and Nie, Liqiang and Liu, Yebin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
