# Non-Synergistic VAE
Pytorch implementation of Non-Synergistic Variational Autoencoders. Abstract: http://bit.ly/2RG0ZVe 

This work is the result of the dissertation of my M.Sc. Machine Learning from UCL.
It has been accepted to:
<ul style="list-style-type:disc">
  <li><b>Neural Information Processing Systems 2018 Workshop "LatinX in AI Research". Poster and Oral presentation</b></li>
</ul>
<br>

### Dependencies
```
python 3.6
pytorch 0.4.1
numpy 1.12
visdom 0.1.8.5
```

### Datasets
2D Shapes(dsprites) Dataset
You can download from: https://github.com/deepmind/dsprites-dataset
Place the file inside the main directory
```
.
└── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
└── ...
```

### Usage
you can reproduce results below as follows

```
python run.py --alpha=5.0 --omega=0.9 --metric="Test_gt" --steps=150000 --gt-interval=5000
```
If you want to visualise the plots in real time:

initialize visdom
```
python -m visdom.server
```

check training process on the visdom server (change local host to the IP address if using cloud services)
```
localhost:8097
```
Run the following commands:
```
python run.py --alpha=5.0 --omega=0.9 --metric="test" --steps=150000 
```
<br>

### Results - 2D Shapes(dsprites) Dataset

#### Latent Space Traverse
<p align="center">
<img src=misc/traversals.gif>
</p>
<br>

#### Comparison with Factor VAE
<p align="center">
<img src=misc/latents.png>
</p>

#### Plots of losses of first 4k steps
<p align="center">
<img src=misc/plots.png>
</p>

