# Data Documentation

We provide a scripts for rendering training data from human scans, many thanks [Ruizhi Shao](https://dsaurus.github.io/saurus/) for sharing this code. Take [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) as an example. 

- Download THuman2.0 scan data from [This Link](https://github.com/ytrock/THuman2.0-Dataset) and the SMPL-X fitting parameters from [This Link](https://drive.google.com/file/d/1rnkGomScq3yxyM9auA-oHW6m_OJ5mlGL/view?usp=sharing). Then spilt the THuman2.0 scans into train set and validation set. We use theSMPL-X parameters to normalization the orientation of human and it is not essential. Comment L133-L140 in [render_data.py](render_data.py#L133-L140) if you do not need to normalization the orientation of human scans in THuman2.0.
```
./Thuman2.0
├── THuman2.0_Smpl_X_Paras/
├── train/
│   ├── 0004/
│   │   ├── 0004.obj
│   │   ├── material0.jpeg
│   │   └── material0.mtl
│   ├── 0005
│   ├── 0007
│   └── ...
└──val
    ├── 0000
    ├── 0001
    ├── 0002
    └── ...
```

- Set the correct ```thuman_root``` and ```save_root``` in [render_data.py](render_data.py#L214-L215). Reset the ```cam_nums``` , ```scene_radius``` and camera parameters to make the training data similar to your targeting real-world scenario.
