# radio-galaxy-classifier

Radio Galaxy Classifier (RGC). The acronym also alludes to Radha Govinda Chandra (1878–1975), a Bangladeshi-Indian variable star observer.

#### Setup configuration

```bash
$ python -m utils.setup_configs --config downstream
```

#### Train

```bash
$ python train.py --model downstream
```

Cite this work:

```bibtex
@article{HOSSAIN2023601,
title = {Morphological classification of Radio Galaxies using Semi-Supervised Group Equivariant CNNs},
journal = {Procedia Computer Science},
volume = {222},
pages = {601-612},
year = {2023},
note = {International Neural Network Society Workshop on Deep Learning Innovations and Applications (INNS DLIA 2023)},
issn = {1877-0509},
doi = {https://doi.org/10.1016/j.procs.2023.08.198},
url = {https://www.sciencedirect.com/science/article/pii/S1877050923009638},
author = {Mir Sazzat Hossain and Sugandha Roy and K.M.B. Asad and Arshad Momen and Amin Ahsan Ali and M Ashraful Amin and A. K. M. Mahbubur Rahman},
keywords = {Radio Galaxy, Fanaroff-Riley, G-CNN, SimCLR, BYOL, Semi-supervised Learning}
}
```
