<div align="center">
<h2>
Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift

<a href="https://imagine.enpc.fr/~elliot.vincent/">Elliot Vincent</a>&emsp;
<a href="https://www.di.ens.fr/~ponce/">Jean Ponce</a>&emsp;
<a href="https://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>

<p></p>

</h2>
</div>

Official PyTorch implementation of [**Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift**](https://github.com/ElliotVincent/SitsSCD).
Check out our [**webpage**](https://imagine.enpc.fr/~elliot.vincent/) for other details!

![alt text](https://github.com/ElliotVincent/SitsSCD/blob/main/sits_scd.png?raw=true)

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
python train.py dataset=<dynamiceathnet or muds> mode=<train or eval>
```

## Bibliography

[1] Adam Van Etten et al. *The multitemporal urban development spacenet dataset*. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6398–6407, 2021.

[2] Aysim Toker et al. *Dynamicearthnet: Daily multi-spectral satellite dataset for semantic change segmentation*.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21158–21167, 2022.