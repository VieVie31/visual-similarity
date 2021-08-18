## Learning an adaptation function to assess image visual similarities

<p align="center">
  <img width="500" alt="Pipeline to learn an adaptation function able to compute visual similarity from image pairs."
                   src="https://user-images.githubusercontent.com/18449334/120085500-03da3380-c0d9-11eb-8cf8-aaf54d399a66.png">
</p>

Please cite for the main method :

```
@inproceedings{risser2021learning,
  title={Learning an adaptation function to assess image visual similarities},
  author={Risser-Maroix, Olivier and Kurtz, Camille and Lomenie, Nicolas},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2498--2502},
  year={2021},
  organization={IEEE}
}
```


The extended version contains better scores by benchmarking 37 pretrained features extractors :

```
@article{risser2022learning,
  title={Learning an Adaptation Function to Assess Image Visual Similarities},
  author={Risser-Maroix, Olivier and Marzouki, Amine and Djeghim, Hala and Kurtz, Camille and Lomenie, Nicolas},
  journal={arXiv preprint arXiv:2206.01417},
  year={2022}
}
```

<br>


### Quick training 

Open the `example.ipynb` notebook for fast explaination…



### Want to use the same TLL_obj split ?

`TLL_obj.csv` contains the name of the images composing the TLL_obj presented in our paper…

To compare score with use you should use the exact same split for fairness.


🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧

### Pretrained Model

<table>
  <tr>
    <th>epochs</th>
    <th>Avg top1 accuracy</th>
    <th>Avg top5 accuracy</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>150</td>
    <td>39.39%</td>
    <td>N/A</td>
    <td><a href="#">full checkpoint</a></td>
    <td><a href="#">train logs</a></td>
    <td><a href="#">val logs</a></td>
  </tr>
</table>

You can download the full checkpoint, which contains the weights of the adaptation module and the state of the optimizer.

<br>

### Clip extracted features

<table>
  <tr>
    <th>Concatenated features size</th>
    <th>Reduced features with PCA size</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>5440</td>
    <td>256</td>
    <td><a href="#">concatenated features</a></td>
    <td><a href="#">reduced features</a></td>
    <td><a href="#">serialized sklearn pca object</a></td>
  </tr>
</table>

You can download the extracted features from the clip Resnet50x4 model.

<br>

### Training an adaptation module

When you have the extracted embeddeings in serialized form, you can train an adaptation module of your choice using the following command:

```
python train.py -d PATH/TO/EMBDS.NPY --model original -s PATH/WHERE/TO/SAVE/TENSORBOARD_METRICS -e NUM_EPOCHS -r NUM_RUNS -t TEMPERATURE --test-split SPLITTING_PERCENTAGE -k TOPK_VALS --gpu
```

To know more either check the train [doc](docs/train.html) or run:
```
python train.py -h
```

In order to have our results run the following command:

```
python train.py -d PATH/TO/CLIP_EMBDS.NPY --model original -s PATH/WHERE/TO/SAVE/TENSORBOARD_METRICS -e 150 -r 1 -t 15 --test-split 0.25 -k 1 3 5 --gpu
```

<br>

### Extracting features 

To extract the embeddings of a dataset, using the models defined in [models.py](featuresExtractor/models.py), you can run the following command:

```
python extract.py -d PATH/TO/DATASET -s PATH/WHERE/TO/SAVE/THE/EMBEDDEINGS -b BATCH_SIZE --data-aug OR --no-data-aug --processor PROCESSOR
```

To see all the implemented processors, see the following [documentation](docs/featuresExtractor/processor.html).

In our case, we [augment](featuresExtractor/transforms.py) the images and use the adaptation processor with pca, this can be done by running:

```
python extract.py -d PATH/TO/DATASET -s PATH/WHERE/TO/SAVE/THE/EMBEDDEINGS -b 2048 --data-aug --processor adapt-pca 256
```

<br>




