# MSER-SGPO
Weakly Supervised Instance Segmentation Using Multi-stage Erasing Refinement and Saliency-guided Proposals Ordering
## PyTorch Implementation
The pytorch  contains:
* the training code.
* the inference code.
### Prerequisites
* System (tested on Ubuntu 14.04LTS)
* Python>=3.5
* PyTorch>=0.4
* pycharm or other IDE
### Run demo
1. download the PASCAL-VOC2012 dataset:
```Bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
2. download the object mask proposals:
https://cvlsegmentation.github.io/cob
3. download the saliency map:
https://github.com/zijundeng/R3Net
4. run the train code
```Bash
prm_test.py
```
5 run the inference code
```Bash
fu.py
```
