
<div align="center">

# [ICCV 2023] Generalized Lightness Adaptation with Channel Selective Normalization.

[Mingde Yao](https://scholar.google.com/citations?user=fsE3MzwAAAAJ&hl=en)\*, [Jie Huang](https://huangkevinj.github.io/)\*, [Xin Jin](http://home.ustc.edu.cn/~jinxustc/), [Ruikang Xu](https://scholar.google.com/citations?user=PulrrscAAAAJ&hl=en), Shenglong Zhou, [Man Zhou](https://manman1995.github.io/), [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/)

University of Science and Technology of China (USTC)


[[`Paper`](http://home.ustc.edu.cn/~mdyao/ICCV_2023_CS_Norm.pdf)] [[`BibTeX`](#heart-citing-us)] :zap: :rocket: :fire:

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

:rocket: Welcome! This is the official repository of [ICCV'23] Generalized Lightness Adaptation with Channel Selective Normalization. 

</div>



## 📌 Overview

>Lightness adaptation is vital to the success of image processing to avoid unexpected visual deterioration, which covers multiple aspects, e.g., low-light image enhancement, image retouching, and inverse tone mapping. Existing methods typically work well on their trained lightness conditions but perform poorly in unknown ones due to their limited generalization ability. To address this limitation, we propose a novel generalized lightness adaptation algorithm that extends conventional normalization techniques through a channel filtering design, dubbed Channel Selective Normalization (CSN). The proposed CSN purposely normalizes the statistics of lightness-relevant channels and keeps other channels unchanged, so as to improve feature generalization and discrimination. To optimize CSN, we propose an alternating training strategy that effectively identifies lightness-relevant channels. The model equipped with our CSN only needs to be trained on one lightness condition and can be well generalized to unknown lightness conditions. Experimental results on multiple benchmark datasets demonstrate the effectiveness of CSN in enhancing the generalization ability for the existing lightness adaptation methods. 


![image](https://github.com/mdyao/CSN-Net-ICCV2023/assets/33108887/f4c9b327-51fa-4832-8069-ab6919100277)

Overview of our proposed method. (a) Channel selective normalization (CSN), which consists of an instance-level normalization module and a differential gating module. (b) Differential gating module. It outputs a series of on-off switch gates for binarized channel selection in CSN. (c) Alternating training strategy. In the first step, we optimize the parameters outside the CSN to keep an essential ability for lightness adaptation. In the second step, we only update the parameters inside the CSN  (see (a)\&(b)) with lightness-perturbed images. The two steps drive the CSN to select channels sensitive to lightness changes, which are normalized in $x_{n+1}$. 


<!--
## :star: News

* August. 18, 2023: We release the core code of our algorithm.
<!--, including the implementation of network architecture and training strategy.  ### Highlights
* More models and configurations will be open source soon, we need some time to organize our data and code. -->



##  :sunflower: Results

![image-20230817235035699](README/image-20230817235035699.png)

Visual comparisons of the generalized image retouching on the MIT-Adobe FiveK  dataset. The models are trained on the original dataset and tested on the scaled lightness condition.  



## :rocket: Usage

<!-- This repository is the **official implementation** of the paper, "Generalized Lightness Adaptation with Channel Selective Normalization", where more implementation details are presented. -->


More details will be updated recently. 


## :heart: Citing Us
If you find this repository or our work useful, please consider giving a star :star: and citation :t-rex:, which would be greatly appreciated:

```bibtex
@inproceedings{yao2023csnnet,
	title={Generalized Lightness Adaptation with Channel Selective Normalization},
	author={Mingde Yao, Jie Huang, Xin Jin, Ruikang Xu, Shenglong Zhou, Man Zhou, and Zhiwei Xiong},
	booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	year={2023}
}
```


## :email: Contact

<!-- If you have any problem with the released code, please do not hesitate to open an issue.-->

For any inquiries or questions, please contact me by email (mdyao@mail.ustc.edu.cn) or [WeChat](https://user-images.githubusercontent.com/33108887/225539514-7c10ccc7-0710-4d7a-8a09-643cf3832d53.png). 
