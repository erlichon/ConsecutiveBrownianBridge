# Frame Interpolation with Consecutive Brownian Bridge

<div align="center">
  
[Zonglin Lyu](https://zonglinl.github.io/), [Ming Li](https://liming-ai.github.io/), [Jianbo Jiao](https://jianbojiao.com/), [Chen Chen](https://www.crcv.ucf.edu/chenchen/)

[![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)](https://zonglinl.github.io/videointerp/) [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://youtu.be/X3xcYm-qajM)  [![arXiv](https://img.shields.io/badge/arXiv-2405.05953-00ff00.svg)](https://arxiv.org/abs/2405.05953)

</div>

<p align="center">
<img src="images/Teaser.jpg" width=95%>
<p>

## Overview
We takes advangtage of optical flow estimation in the autoencoder part and design the Consecutive Brownian Bridge Diffusion that transits among three frames specifically for the frame interpolation task. (a)**The autoencoder with flow estimation** improves the visual quality of frames decoded from the latent space. (b) **The Consecutive Brownian Bridge Diffusion** reduce cumulative variance during sampling, which is prefered in VFI becuase there is a *deterministic groundtruth* rather than *a diverse set of images*. (c) During inference, the decoder recieves estimated latent features from the Consecutive Brownian Bridge Diffusion.

<p align="center">
<img src="images/overview.jpg" width=95%>
<p>

## Quantitative Results
Our method achieves state-of-the-art performance in LPIPS/FloLPIPS/FID among all recent SOTAs. 
<p align="center">
<img src="images/quant.png" width=95%>
<p>

## Qualitative Results
Our method achieves state-of-the-art performance in LPIPS/FloLPIPS/FID among all recent SOTAs. 
<p align="center">
<img src="images/qualadd-1.png" width=95%>
<p>

For more visualizations, please refer to our <a href="https://zonglinl.github.io/videointerp/">project page</a>.

## Inference
**Please comment line 6 in ```utils.py``` before we provide training codes!**

Please install necessary packages in requirements.txt. 

**Please leave the *load_VFI* and *ckpt_path* in the config file as empty**, otherwise you need to download the model weights of VFIformer from <a href="https://drive.google.com/drive/folders/140bDl6LXPMlCqG8DZFAXB3IBCvZ7eWyv"> here</a> and VQ Model from the link below. You need to change the path of load_VFI and ckpt_path to the path of downloaded weights, then run:

```
python interpolate.py --resume_model path_to_model_weights --frame0 path_to_the_previous_frame --frame1 path_to_the_next_frame
```
This will interpolate 7 frames in between, you may modify the code to interpolate different number of frames with a bisection like methods
The weights of of our trained model can be downloaded <a href="https://drive.google.com/file/d/1Z5kPMdYiC4CSvl1mrQLz9MqtJx7RjvrK/view?usp=sharing">here</a>. The VQ Model is available <a href="https://drive.google.com/file/d/1V8WS7bZe_RTCtyYZ6ZFkur8sHbKvVkT8/view?usp=sharing"> here</a>

## Training and Evaluating

This part will be released after paper is accepted

## Acknowledgement

We greatfully appreaciate the source code from [BBDM](https://github.com/xuekt98/BBDM), [LDMVFI](https://github.com/danier97/LDMVFI), and [VFIformer](https://github.com/dvlab-research/VFIformer)

## Citation

If you find this repository helpful for your research, please cite:

```
@misc{lyu2024frame,
      title={Frame Interpolation with Consecutive Brownian Bridge Diffusion}, 
      author={Zonglin Lyu and Ming Li and Jianbo Jiao and Chen Chen},
      year={2024},
      eprint={2405.05953},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
