# Positive-Similarity

Recent self-supervised architecture [BYOL](https://arxiv.org/pdf/2006.07733.pdf) provide a way to learn representation from only positive pairs of transformation of a single image.
In this project, we would like to investigate the ability of this kind of architecture to learn in the more general _metric learning_ setting from only positive pairs.
Previous works rely on _triplet networks_ or _siamese networks_.
Those architectures need complex mining procedures to create negative triplet for efficient learning.
In some case, the negative samples could not even be doable (eg. because of the compositionality of attributes).
This project could be beneficial and open new perspectives for _metric learning_ in those complicated cases.

## Questions, Ideas & Research Plan

- [ ] BYOL like architecture re-implementation (on TLL dataset, on VISAGOLY ?)
- [ ] Do we necessarily need two networks (_online_ and _target_ networks surch as in MoCo or BYOL) to learn from only positive pairs ? Can we only rely on a single network and use the [_batchnorm_ trick](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html) to prevent collapse ?
- [ ] other ? (add other ideas if any)
