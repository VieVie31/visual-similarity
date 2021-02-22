# Positive-Similarity

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


