# Deep Generative Model for Spatial-spectral Unmixing with Multiple Endmember Priors (DGMSSU)

The code in this toolbox implements the "[Deep Generative Model for Spatial-spectral Unmixing with Multiple Endmember Priors](https://ieeexplore.ieee.org/document/9759362)". More specifically, it is detailed as follow.

<center><img src="./fig/fig1.png" width="80%"><center>
<center><img src="./fig/fig2.png" width="60%"><center>
  
## Train

#### Step 1: Extract endmember bundles

Run `python extraceEMBundles.py`

#### Step 2: Segment the image into superpixels for DGMGCN and DGMSAN

Run `python SLIC_preprocess.py`

#### Step 3: Train DGMSSU

Run `python main_pytorch_CNN.py` to train the DGMCNN.

Run `python main_pytorch_GCN.py` to train the DGMGCN.

Run `python main_pytorch_SelfAttention.py` to train the DGMSAN.

## Result
  
The unmixing results will be saved at `./model_torch/DGMSSU/out/`.


## Visualization

Run `python test.py` to get  visual results.





## Citation

**Please kindly cite the papers if this code is useful and helpful for your research.**

```
@ARTICLE{shi2022Deep,  
author={Shi, Shuaikai and Zhang, Lijun and Altmann, Yoann and Chen, Jie},  
journal={IEEE Transactions on Geoscience and Remote Sensing},   
title={Deep Generative Model for Spatial-spectral Unmixing with Multiple Endmember Priors},   
year={2022},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/TGRS.2022.3168712}}
```

## Licensing

Copyright (C) 2022 Shuaikai Shi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

## Contact Information:

If you encounter any bugs while using this code, please do not hesitate to contact us.

Shuaikai Shi [_shuaikai_shi@mail.nwpu.edu.cn](_shuaikai_shi@mail.nwpu.edu.cn) is with the Center of Intelligent Acoustics and Immersive Communications, School of Marine Science and Technology, Northwestern Polytechinical University, Xi’an 710072, China
