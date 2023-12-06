<div align="center">

# <b>GPS-Gaussian</b>: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis

[Shunyuan Zheng](https://shunyuanzheng.github.io)<sup>1</sup>, [Boyao Zhou](https://morpheo.inrialpes.fr/people/zhou)<sup>2</sup>, [Ruizhi Shao](https://dsaurus.github.io/saurus)<sup>2</sup>, [Boning Liu](https://scholar.google.com/citations?user=PG1mUewAAAAJ)<sup>2</sup>, [Shengping Zhang](http://homepage.hit.edu.cn/zhangshengping)<sup>1</sup>, [Liqiang Nie](https://liqiangnie.github.io)<sup>1</sup>, [Yebin Liu](https://www.liuyebin.com)<sup>2</sup>

<sup>1</sup>Harbin Institute of Technology <sup>2</sup>Tsinghua Univserity

### [Projectpage](https://shunyuanzheng.github.io/GPS-Gaussian) · [Paper](https://arxiv.org/pdf/xxx.pdf) · [Video](https://youtu.be/TBIekcqt0j0)

</div>

<img src="https://shunyuanzheng.github.io/assets/GPS-Gaussian/images/teaser.png">

***Abstract**: We present a new approach, termed GPS-Gaussian, for synthesizing novel views of a character in a real-time manner. The proposed method enables 2K-resolution rendering under a sparse-view camera setting. Unlike the original Gaussian Splatting or neural implicit rendering methods that necessitate per-subject optimizations, we introduce Gaussian parameter maps defined on the source views and regress directly Gaussian Splatting properties for instant novel view synthesis without any fine-tuning or optimization. To this end, we train our Gaussian parameter regression module on a large amount of human scan data, jointly with a depth estimation module to lift 2D parameter maps to 3D space. The proposed framework is fully differentiable and experiments on several datasets demonstrate that our method outperforms state-of-the-art methods while achieving an exceeding rendering speed.*

Code is coming soon.

## Free View Rendering Results
### Data collected by ourselves

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/36d06407-fadc-485a-864b-961fbd4d4b60

### Data from [DNA-Rendering](https://dna-rendering.github.io/)

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/d392673c-13cd-442d-aa94-6629c9edfb3c

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/371171ca-46a9-427b-9549-9d65fc4b135d

## Live Demo

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/e5f30839-ab01-4594-81e8-f1ea5314ff36

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/31fd4012-7344-418c-aa06-a614ded3c236

https://github.com/ShunyuanZheng/GPS-Gaussian/assets/33752042/fa894569-1771-438e-927d-7680147f17ca



