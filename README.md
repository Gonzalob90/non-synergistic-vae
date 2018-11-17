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

Dsprites is a dataset of 2d shapes generated from 6 ground truth independent latent factors: color, shape, scale, rotation, x and y positions. There are a total of 737280 examples generated from the combination of the values of each of these factors:

* Color: white
* Shape: square, ellipse, heart
* Scale: 6 values linearly spaced in [0.5, 1]
* Orientation: 40 values in [0, 2 pi]
* Position X: 32 values in [0, 1]
* Position Y: 32 values in [0, 1]

#### Latent Space Traverse

We show below the reconstructions resulting from the traversal of each latent z_i over three standard deviations around the unit Gaussian prior mean while keeping the remaining 9/10 latent units fixed to the values obtained by running inference on an image from the dataset. Row 1 is the original, row 2 is the reconstruction, and row 3 to 12 represent each latent z_i. In this gif, we print images every 2000 steps up to 110k steps. As you can see, the model learns the scale, rotation (2), x and y axis in an unsupervised setting. 

<br>
<p align="center">
<img src=misc/traversals.gif>
</p>
<br>

#### Comparison with Factor VAE

We compare our model with Factor VAE. On the left, we show the latent traversals (110k steps) for Non-Syn VAE and Factor VAE. We put the name of the learned latent for an easier visualisation. On the right, we show the mean activations of the latents, which are figures that represent the values of the learned latent averaged across the other variables. For example, for Non-Syn VAE, z2 and z5 represent the "y" axis and "x" axis respectevely; we see in the column name 'pos' the representation of the mean activation of each latent zi as a function of all 32x32 location averaged across objects, rotations and scales. Since this representation is disentangled, we should see activations in the column of positions and no interaction in the scale and rotation columns. Likewise, z3 and z6 show the mean activation as a function of roration averaged across rotations and scale, whereas the row z6 shows the mean activation as a function of the scale averaged across rotations and positions. The three coloured lines in the scale and rotation columns are related to the objects "square, ellipse, heart". In the position column, the red colour represents higher values. The latent z7 is non-informative.

<p align="center">
<img src=misc/latents.png>
</p>

In addition, we plot the latent traversal for the original VAE [1] just to showcase the differences of having a disentangled representation. On the left, you can see clearly that most that model didn't disentangle any of ground truth factors, since each of the latents has information about more than one factors. On the left, we see the mean activations which again represent clearly that most of the latents have information about position, scale and rotation.

<p align="center">
<img src=misc/VAE_traversals.png>
</p>

#### Plots of losses of first 4k steps

We plot the Reconstruction loss, the KL loss and the synergy loss for the first 4000 steps.

<p align="center">
<img src=misc/plots.png>
</p>

#### Latent space Traverse of CelebA

We show below the trasverse of latents for the dataset CelebA ( http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html ) using the same procedure as for the dsprites dataset. In terms of implementation, the difference between this CelebA and dsprites was that for the latter we used a Cross-Entropy loss for the reconstruction term whereas for the former one we used a Mean Square Error Loss. 

<p align="center">
<img src=misc/faces_celeba.png>
</p>

### References

[1] D. P. Kingma and M. Welling. Auto-encoding variational bayes. ICLR, 2014.

