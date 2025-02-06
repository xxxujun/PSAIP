# PSAIP: Prior Structure-Assisted Identity-Preserving Network for Face Animation

### Installation

We support ```python3```.(Recommended version is Python 3.9).
To install the dependencies run:
```bash
pip install -r requirements.txt
```


### YAML configs
 
There are several configuration files one for each `dataset` in the `config` folder named as ```config/dataset_name.yaml```. 

See description of the parameters in the ```config/taichi-256.yaml```.

### Datasets

 **VoxCeleb**. Follow instructions from [video-preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing). 


### Training
To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1 python run.py --config config/dataset_name.yaml --device_ids 0,1
```
A log folder named after the timestamp will be created. Checkpoints, loss values, reconstruction results will be saved to this folder.



### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config config/dataset_name.yaml --checkpoint '{checkpoint_folder}/checkpoint.pth.tar'
```
The `reconstruction` subfolder will be created in `{checkpoint_folder}`.
The generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
To compute metrics, follow instructions from [pose-evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).


### Image animation demo
- notebook: `demo.ipynb`, edit the config cell and run for image animation.
- python:
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./source.jpg --driving_video ./driving.mp4
```

# Acknowledgments
The main code is based upon [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [TPSMM](https://github.com/snap-research/articulated-animation)

Thanks for the excellent works!

And Thanks to:

- [@chenxwh](https://github.com/chenxwh): Add Web Demo & Docker environment [![Replicate](https://replicate.com/yoyo-nb/thin-plate-spline-motion-model/badge)](https://replicate.com/yoyo-nb/thin-plate-spline-motion-model) 

- [@TalkUHulk](https://github.com/TalkUHulk): The C++/Python demo is provided in [Image-Animation-Turbo-Boost](https://github.com/TalkUHulk/Image-Animation-Turbo-Boost)

- [@AK391](https://github.com/AK391): Add huggingface web demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CVPR/Image-Animation-using-Thin-Plate-Spline-Motion-Model)