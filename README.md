# RLCFE-Net

Implementation of paper - [RLCFE-Net: A Reparameterization Large Convolutional Kernel
Feature Extraction Network for Weed Detection in Multiple Scenarios]()

<div align="center">
    <a href="./">
        <img src="./figure/fig1.png" width="79%"/>
    </a>
</div>


## Performance 

Sixweeds


| Model               | Parameters             | FLOPs (G)           | Size (MB)          | F1-core            | Fps                | mAP(50)           | mAP(50:95)        |
| :------------------ | :--------------------: | :------------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
| **YOLOv5m**        | <u>20891523</u>       | <u>48.3</u>         | <u>42.2</u>       | 0.897              | **59.523**         | 0.939              | 0.776              |
| **YOLOv6m**        | 34810000               | 85.64                | 72.5              | 0.904              | 41.684             | 0.956              | 0.832              |
| **YOLOv7**         | 37223526               | 105.2                | 74.8              | 0.916              | 28.248             | 0.947              | 0.816              |
| **YOLOv8m**        | 25859794               | 79.1                 | 52.0              | 0.945              | 38.167             | 0.965              | 0.853              |
| **ASPPF-YOLOv8m**  | 36312466               | 87.2                 | 72.9              | <u>0.950</u>       | 36.900             | 0.963              | 0.852              |
| **LS-YOLOv8m**     | 25715420               | 77.2                 | 51.8              | 0.947              | 35.460             | 0.966              | 0.861              |
| **LW-YOLOv8m**     | **9883660**            | **38.0**             | **20.2**          | 0.934              | <u>43.478</u>      | 0.964              | 0.849              |
| **YOLOv9c**        | 25441698               | 103.2                | 48.9              | **0.952**          | 30.769             | <u>0.971</u>       | <u>0.878</u>       |
| **GELANc**         | 25441698               | 103.2                | 51.4              | 0.941              | 30.395             | 0.963              | 0.872              |
| **YOLOv10l**       | 25774580               | 127.2                | 52.2              | 0.917              | 26.525             | 0.944              | 0.829              |
| **RLCAFE-Net**     | 23134498               | 62.1                 | 46.9              | 0.947              | 39.370             | **0.977**          | **0.881**          |


<!-- small and medium models will be released after the paper be accepted and published. -->


## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name RLCFE-Net -it -v your_coco_path/:/coco/ -v your_code_path/:/RLCFE-Net --shm-size=64g nvcr.io/nvidia/pytorch:21.11-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /RLCFE-Net
```

</details>


## Evaluation


``` shell
# evaluate dr-gelan models
# python val.py --data data/mycoco128.yaml --img 640 --batch 4 --conf 0.001 --iou 0.7 --device 0 --weights './RLCFE-Net-c.pt' --save-json --name RLCFE-Net_c_640_val
```


## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

*Download [Sixweeds dataset](https://github.com/guoao784/Sixweeds) images and labels.

Single GPU training

``` shell
# train RLCFE-Net models
# python train.py --workers 8 --device 0 --batch 4 --data data/mycoco128.yaml --img 640 --cfg models/detect/RLCFE-Net-c.yaml --weights '' --name RLCFE-Net-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
```

Multiple GPU training

``` shell
# train RLCFE-Net models
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch 128 --data data/mycoco128.yaml --img 640 --cfg models/detect/RLCFE-Net-c.yaml --weights '' --name RLCFE-Net-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
```


## Re-parameterization

Under construction.


## Citation

```
1:@article{
  title={RLCFE-Net: A Reparameterization Large Convolutional Kernel
Feature Extraction Network for Weed Detection in Multiple Scenarios},
  author={Ao Guo, Zhenhong Jia, Baoquan Ge, Wei Chen, Sensen Song, Congbing He, Gang Zhou, Jiajia Wang, Xiaoyi Lv},
  journal={Journal of Advanced Research},
  year={2024}
}
```
```
Paper is currently under review. ;)
```
```
/2:@article{2024
  title={A lightweight weed detection model with global contextual joint features},
  author={Ao Guo, Zhenhong Jia, Jianyi Wang, Jiajia Wang , Gang Zhou, Baoquan Ge and Wei Chen},
  journal={Engineering Applications of Artificial Intelligence},
  year={2024}
}
```
```
https://doi.org/10.1016/j.engappai.2024.108903
```

```
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint},
  year={2024}
}
```

```
@article{chang2023yolor,
  title={{YOLOR}-Based Multi-Task Learning},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard Robert and Chou, Gene and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2309.16921},
  year={2023}
}
```


## Teaser

Parts of code of [A lightweight weed detection model with global contextual joint features](https://doi.org/10.1016/j.engappai.2024.108903) and [Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616) are released in the repository.



## Acknowledgements

<details><summary> <b>Expand</b> </summary>
    
* [https://github.com/guoao784/DR-GELAN](https://github.com/guoao784/RLCFE-Net)
* [https://github.com/guoao784/Sixweeds](https://github.com/guoao784/Sixweeds)
* [https://github.com/guoao784/LW-YOLOv8](https://github.com/guoao784/LW-YOLOv8)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details>
