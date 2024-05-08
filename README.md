# Frame Interpolation with Consecutive Brownian Bridge

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-xxxx.xxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxx)&nbsp;
</div>

<p align="center" style="font-size:1em;">
  <a href="https://zonglinl.github.io/videointerp/">Frame Interpolation with Consecutive Brownian Bridge</a>
</p>

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
<img src="images/quant.jpg" width=95%>
<p>

## Qualitative Results
Our method achieves state-of-the-art performance in LPIPS/FloLPIPS/FID among all recent SOTAs. 
<p align="center">
<img src="images/qualadd-1.jpg" width=95%>
<p>

For more visualizations, please refer to our <a href="https://zonglinl.github.io/videointerp/">project page</a>.

## Inference

Please install necessary packages in requirements.txt, then run:

```
python interpolate.py --resume_model path_to_model_weights --frame0 path_to_the_previous_frame --frame1 path_to_the_next_frame
```
This will interpolate 7 frames in between, you may modify the code to interpolate different number of frames with a bisection like methods
The weights of of our trained model can be downloaded <a href="https://zonglinl.github.io/videointerp/">here</a>.

## Training and Evaluating

This part will be released after paper is accepted
