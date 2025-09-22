
# GAVA: Geometric-Aware Visual Attention for Image Captioning

This repository is for **GAVA: Geometric-Aware Visual Attention for Image Captioning**. The original paper can be found [here](https://www.sciencedirect.com/science/article/pii/S1051200425004579?via%3Dihub).

Please cite with the following BibTeX:

### Journal Citation (Digital Signal Processing 2025):
```bibtex
@article{HOSSAIN2025105435,
  author = {Hossain, Mohammad Alamgir and Ye, ZhongFu and Hossen, Md. Bipul and Rahman, Md. Atiqur and Islam, Md. Shohidul and Abdullah, Md. Ibrahim},
  title = {GAVA: Spatial Awareness in Image Captioning with Geometric-Aware Visual Attention},
  journal = {Digital Signal Processing},
  year = {2025},
  volume = {105435},
  issn = {1051-2004},
  doi = {https://doi.org/10.1016/j.dsp.2025.105435},
  url = {https://www.sciencedirect.com/science/article/pii/S1051200425004579}
}
```

<p align="center">
  <img src="images/framework.jpg" width="800"/>
</p>

## Requirements
* Python 3
* CUDA 10
* numpy
* tqdm
* easydict
* [PyTorch](http://pytorch.org/) (>1.0)
* [torchvision](http://pytorch.org/)
* [coco-caption](https://github.com/ruotianluo/coco-caption)

## Data preparation
1. Download the [bottom up features](https://github.com/peteanderson80/bottom-up-attention) and convert them to npz files:
```
python2 tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_100
```

This command will generate two different types of features:
- `up_down_100`: Contains the main feature files.
- `up_down_100_box`: Contains the bounding box coordinates.

2. Download the [annotations](https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS) into the mscoco folder. More details about data preparation can be referred to [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).

3. Download [coco-caption](https://github.com/ruotianluo/coco-caption) and set up the path of `__C.INFERENCE.COCO_PATH` in `lib/config.py`.

4. The pretrained models and results for Cross Entropy Loss can be downloaded [here](https://drive.google.com/file/d/1_5FttnUADK49ZW1fUEdZn-kitgLgytkS/view?usp=drive_link) and [here](https://drive.google.com/file/d/1q0qSYpHfPvx-OhZPNXnp7Ko6b0eEOstc/view?usp=drive_link).

5. The pretrained models and results for CIDEr Score Optimization can be downloaded [here](https://drive.google.com/file/d/1akyfuiCclsU12nnZEWJpBronykBQd5ZN/view?usp=drive_link) and [here](https://drive.google.com/file/d/1RkU9i8Ow70ps-103OSj6O24_4ZV3wgv6/view?usp=drive_link).

6. The geometric features can be downloaded [here](https://drive.google.com/file/d/1eqmeavgomteESeR43tpW8awIl0qOcDiA/view?usp=drive_link).

### Generating and Normalizing Geometric Features
To generate and normalize geometric features, follow these steps:

1. Ensure you have extracted the bounding box coordinates into `up_down_100_box` as described above.

2. Use the provided script to generate and normalize the geometric features, which will be saved in the `geo_feats` directory.

## Training
### Train GAVA model
```
bash experiments/gava/train.sh
```

### Train GAVA model using self critical
Copy the pretrained model into `experiments/gava_rl/snapshot` and run the script:
```
bash experiments/gava_rl/train.sh
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python3 main_test.py --folder experiments/model_folder --resume model_epoch
```

## Acknowledgements
We would like to thank the following contributions and resources that made this work possible:

- [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) for providing the training framework and evaluation tools.  
- The [PyTorch](http://pytorch.org/) team for their excellent deep learning library.  
- [X-Linear Attention Networks for Image Captioning (X-LAN)](https://github.com/JDAI-CV/image-captioning) [Pan et al., CVPR 2020], which our implementation builds upon and extends with geometric-aware visual attention.


