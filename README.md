<div align="center">
<h2>
Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift

<a href="https://imagine.enpc.fr/~elliot.vincent/">Elliot Vincent</a>&emsp;
<a href="https://www.di.ens.fr/~ponce/">Jean Ponce</a>&emsp;
<a href="https://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>

<p></p>

</h2>
</div>

Official PyTorch implementation of [**Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift**](https://arxiv.org/abs/2407.07616).
Check out our [**webpage**](https://imagine.enpc.fr/~elliot.vincent/sitsscd) for other details!

We tackle the satellite image time series semantic change detection (SITS-SCD) task with our multi-temporal version of the UTAE [3]. Our model is able to leverage
long range temporal information and provides significant performance boost for this task compared to single- or bi-temporal SCD methods.
We evaluate on DynamicEarthNet [1] and MUDS [2] datasets that exhibit global and multi-year coverage using the SCD metrics defined in [1].

![alt text](https://github.com/ElliotVincent/SitsSCD/blob/main/sitsscd_teaser.png?raw=true)

If you find this code useful, don't forget to <b>star the repo :star:</b>.


## Installation :gear:

### 1. Clone the repository in recursive mode

```
git clone git@github.com:ElliotVincent/SitsSCD.git --recursive
```

### 2. Download the datasets

We use processed versions of the SITS-SCD datasets DynamicEarthNet [1] and MUDS [2]. Our pre-processing consists in image 
compression for memory efficiency. You can download the datasets using the code below or by following these links for 
[DynamicEarthNet](https://drive.google.com/file/d/1cMP57SPQWYKMy8X60iK217C28RFBkd2z/view?usp=drive_link) (7.09G) and
[MUDS](https://drive.google.com/file/d/1RySuzHgQDSgHSw2cbriceY5gMqTsCs8I/view?usp=drive_link) (245M).

```
cd SitsSCD
mkdir datasets
cd datasets
gdown 1RySuzHgQDSgHSw2cbriceY5gMqTsCs8I
unzip Muds.zip
gdown 1cMP57SPQWYKMy8X60iK217C28RFBkd2z
unzip DynamicEarthNet.zip
```

### 3. Create and activate virtual environment

```
conda create -n sitsscd pytorch=2.0.1 torchvision=0.15.2 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate sitsscd
pip install -r requirements.txt
```
This implementation uses PyTorch, PyTorch Lightning and Hydra.

## How to use :rocket:

For both datasets, there are two validation and two test loaders, to account for the presence
or not of spatial domain shift. 
```
python train.py dataset=<dynamicearthnet or muds> mode=<train or eval>
```

## Citing

```bibtex
@article{vincent2024satellite,
    title = {Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift},
    author = {Vincent, Elliot and Ponce, Jean and Aubry, Mathieu},
    journal = {arXiv},
    year = {2024},
  }
```

## Bibliography

[1] Adam Van Etten et al. *The multitemporal urban development spacenet dataset*. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6398–6407, 2021.

[2] Aysim Toker et al. *Dynamicearthnet: Daily multi-spectral satellite dataset for semantic change segmentation*.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21158–21167, 2022.

[3] Vivien Sainte Fare Garnot et al. *Panoptic segmentation of satellite image time series with convolutional
temporal attention networks*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4872–4881, 2021
