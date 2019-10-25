# InvSFM: Revealing Scenes by Inverting Structure from Motion Reconstructions 

![teaser figure](teaser.png)
**Synthesizing Imagery from a SFM Point Cloud:** From left to right--Top view of a SfM reconstruction of an indoor scene; 3D points projected into a viewpoint associated with a source image; the image reconstructed using our technique; and the source image. </p>

<br/>

This repository contains a reference implementation of the algorithms described in the CVPR 2019 paper [**Revealing Scenes by Inverting Structutre from Motion Reconstructions**](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pittaluga_Revealing_Scenes_by_Inverting_Structure_From_Motion_Reconstructions_CVPR_2019_paper.pdf). This paper was selected as a **Best Paper Finalist** at CVPR 2019. For more details about the project, please visit the main [project page](https://www.francescopittaluga.com/invsfm/index.html).


If you use this code/model for your research, please cite the following paper:
```
@inproceedings{pittaluga2019revealing,
  title={Revealing scenes by inverting structure from motion reconstructions},
  author={Pittaluga, Francesco and Koppal, Sanjeev J and Bing Kang, Sing and Sinha, Sudipta N},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={145--154},
  year={2019}
}
```



## Installation Guide

The code was tested with Tensorflow 1.10, Ubuntu 16, NVIDIA TitanX / NVIDIA 1080ti.

### Step 1: Install dependencies

See `requirements.txt`. The training code depends only on tensorflow. The demos additionally depend on Pillow and scikit-image. 

### Step 2: Download the pre-trained model weights 

Run `$ bash download_wts.sh` to programatically download and untar `wts.tar.gz` (1.24G). Alternatively, manually download `wts.tar.gz` from [here](https://drive.google.com/open?id=1D2uYQmrZaZPngDi1U8aSPoXdzuAnEwhb) and untar it in the root directory of the repo.

### Step 3: Download the demo data

Run `$ bash download_data.sh` to programatically download and untar `data.tar.gz` (11G). Alternatively, manually download `data.tar.gz` from [here](https://drive.google.com/open?id=1StpUiEauckZcxHZeBzoq6L2K7pcB9v3E) and untar it in the root directory of the repo.

### Step 4: Run the demos

```
$ python demo_5k.py 
$ python demo_colmap.py
```
Note: Run `$ python demo_5k.py --help` and `$ python demo_colmap.py --help` to see the various demo options available.

### Step 5: Run the training scripts

```
$ python train_visib.py 
$ python train_coarse.py 
$ python train_refine.py 
```
Note: Run `$ python train_*.py --help` to see the various training options available.


















