## Learning an adaptation function to assess image visual similarities

<p align="center">
  <img width="500" alt="Pipeline to learn an adaptation function able to compute visual similarity from image pairs."
                   src="https://user-images.githubusercontent.com/18449334/120085500-03da3380-c0d9-11eb-8cf8-aaf54d399a66.png">
</p>

```
citation
```

<br>

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

## License

add license ?











<!--
Recent self-supervised architecture [BYOL](https://arxiv.org/pdf/2006.07733.pdf) provide a way to learn representation from only positive pairs of transformation of a single image.
In this project, we would like to investigate the ability of this kind of architecture to learn in the more general _metric learning_ setting from only positive pairs.
Previous works rely on _triplet networks_ or _siamese networks_.
Those architectures need complex mining procedures to create negative triplet for efficient learning.
In some case, the negative samples could not even be doable (eg. because of the compositionality of attributes).
This project could be beneficial and open new perspectives for _metric learning_ in those complicated cases.

## Questions, Ideas & Research Plan

- [ ] Read paper [7, 8] ([SimSiam short video](https://www.youtube.com/watch?v=k-PcMBYQsOY), may help to understand)
- [ ] BYOL like architecture re-implementation (on TLL dataset, on VISAGOLY ?)
- [ ] Do we necessarily need two networks (_online_ and _target_ networks surch as in MoCo or BYOL) to learn from only positive pairs ? Can we only rely on a single network and use the [_batchnorm_ trick](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html) to prevent collapse ?
- [ ] other ? (add other ideas if any)


## References

- [[1](https://arxiv.org/pdf/2006.07733.pdf)] Grill, Jean-Bastien et al. _"Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning."_ Advances in Neural Information Processing Systems. Curran Associates, Inc..
- [[2](https://openreview.net/pdf?id=c5QbJ1zob73)] Yuandong Tian, et al. _"Understanding Self-supervised Learning with Dual Deep Networks."_ (2020).
- [[3](https://arxiv.org/pdf/2010.10241.pdf)] Richemond, Pierre H. et al. _“BYOL works even without batch statistics.”_ ArXiv abs/2010.10241 (2020): n. pag.
- [[4](https://papers.nips.cc/paper/2015/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf)] Sadeghi, Fereshteh et al. _"Visalogy: Answering Visual Analogy Questions."_ Advances in Neural Information Processing Systems. Curran Associates, Inc., 
- [[5](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w39/Rosenfeld_Totally_Looks_Like_CVPR_2018_paper.pdf)] Rosenfeld, Amir et al. _"Totally Looks Like - How Humans Compare, Compared to Machines."_ Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
- [[6](https://www.pnas.org/content/pnas/118/3/e2014196118.full.pdf)] Chengxu Zhuang, et al. _"Unsupervised neural network models of the ventral visual stream"_. Proceedings of the National Academy of Sciences 118. 3(2021): e2014196118.
- [[7](https://arxiv.org/pdf/2011.10566.pdf)] Xinlei Chen, et al. _"Exploring Simple Siamese Representation Learning."_ (2020).
- [[8](https://arxiv.org/pdf/2102.06810.pdf)] Yuandong Tian, et al. _"Understanding self-supervised Learning Dynamics without Contrastive Pairs."_ (2021).


-->
