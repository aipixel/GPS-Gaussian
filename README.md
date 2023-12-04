<div align="center">

# <b>GPS-Gaussian</b>: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis

[Shunyuan Zheng](https://shunyuanzheng.github.io)<sup>1</sup>, [Boyao Zhou](https://morpheo.inrialpes.fr/people/zhou)<sup>2</sup>, [Ruizhi Shao](https://dsaurus.github.io/saurus)<sup>2</sup>, [Boning Liu](https://scholar.google.com/citations?user=PG1mUewAAAAJ)<sup>2</sup>, [Shengping Zhang](http://homepage.hit.edu.cn/zhangshengping)<sup>1</sup>, [Liqiang Nie](https://liqiangnie.github.io)<sup>1</sup>, [Yebin Liu](https://www.liuyebin.com)<sup>2</sup>

<sup>1</sup>Harbin Institute of Technology <sup>2</sup>Tsinghua Univserity

### [Projectpage](shunyuanzheng.github.io/GPS-Gaussian) · [Paper](https://arxiv.org/pdf/xxx.pdf) · [Video](https://youtu.be/TBIekcqt0j0)

</div>

<img src="https://shunyuanzheng.github.io/assets/GPS-Gaussian/images/pipeline.png">

***Abstract**: We present a new approach, termed GPS-Gaussian, for synthesizing novel views of a character in a real-time manner. The proposed method enables 2K-resolution rendering under a sparse-view camera setting. Unlike the original Gaussian Splatting or neural implicit rendering methods that necessitate per-subject optimizations, we introduce Gaussian parameter maps defined on the source views and regress directly Gaussian Splatting properties for instant novel view synthesis without any fine-tuning or optimization. To this end, we train our Gaussian parameter regression module on a large amount of human scan data, jointly with a depth estimation module to lift 2D parameter maps to 3D space. The proposed framework is fully differentiable and experiments on several datasets demonstrate that our method outperforms state-of-the-art methods while achieving an exceeding rendering speed.*

Code is coming soon.
