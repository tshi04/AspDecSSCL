# AspDecSSCL 
A Self-Supervised Contrastive Learning Framework for Aspect Detection

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/tshi04/AspDecSSCL/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/AspDecSSCL/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/AspDecSSCL/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/abs/2009.09107)

This repository is a pytorch implementation for the following AAAI'21 paper:

#### [A Simple and Effective Self-Supervised Contrastive Learning Framework for Aspect Detection](https://arxiv.org/pdf/2009.09107.pdf)
[Tian Shi](http://people.cs.vt.edu/tshi/homepage/home), 
[Liuqing Li](https://scholar.google.com/citations?user=eVG56DkAAAAJ&hl=en), 
[Ping Wang](http://people.cs.vt.edu/ping/homepage/), 
[Chandan K. Reddy](http://people.cs.vt.edu/~reddy/)

## Requirements

- Python 3.6.9
- argparse=1.1
- torch=1.4.0
- sklearn=0.22.2.post1
- numpy=1.18.2

# Dataset

Please download processed dataset from here. Place them along side with AapDecSSCL.

```bash
|--- AspDecSSCL
|--- Data
|    |--- bags_and_cases
|    |--- restaurant
|    |    |--- dev.txt
|    |    |--- test.txt
|    |    |--- train.txt
|    |    |--- train_w2v.txt
|--- cluster_results (results, automatically build)
|--- nats_results (results, automatically build)
```

# Train your model from scratch

### Prepare word and aspect embeddings.

```Train word2vec:``` python3 run.py --task word2vec

```Run Kmeans:``` python3 run.py --task kmeans

```Check Kmeans Keywords``` python3 run.py --task kmeans-keywords

### Self-supervised Learning (Teacher Model)

```SSCL Training``` python3 run.py --task sscl-train

Before validation, you need to perform ```aspect mapping```. There is a file ```aspect_mapping.txt``` in ```nats_results```. For ```general```, please change ```nomap``` to ```none```. Other aspects should use their names. Please check ```test.txt``` to validate the names.

```SSCL validation``` python3 run.py --task sscl-validate

```SSCL testing``` python3 run.py --task sscl-test

```SSCL evaluate``` python3 run.py --task sscl-evaluate

```SSCL teacher``` python3 run.py --task sscl-teacher

```SSCL clean results``` python3 run.py --task sscl-clean

### Student Model

```SSCLS training``` python3 run.py --task student-train

```SSCLS validation``` python3 run.py --task student-validate

```SSCLS testing``` python3 run.py --task student-test

```SSCLS testing``` python3 run.py --task student-evaluate

```SSCLS clean``` python3 run.py --task student-clean

## Use Pretrained Model

Coming Soon.

## Citation

```
@article{shi2020simple,
  title={A Simple and Effective Self-Supervised Contrastive Learning Framework for Aspect Detection},
  author={Shi, Tian and Li, Liuqing and Wang, Ping and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2009.09107},
  year={2020}
}
```






