# IW276SS20P7: 3D Human Action Recognition (Jetson Tx2)

This ist a autonome systems lab exersice.

Diese Arbeit beschäftigt sich mit dem Erkennen von menschlichen Aktionen in einem 2D RGB Videostream. Es können dabei Aktionen für bis zu zwei Personen erkannt werden. Die Erkennung ist echtzeitfähig und für eine Ausführung auf einem Jetson TX2 geeignet.

<p align="center">
  <img src="./screenshot.png" width="50%"/>
  <br />
  <a href="https://youtu.be/D3UkoIC3oLU">Link to Demo Video on Youtube</a>
</p>

> This work was done by Janek Gass, Hoang Hai Tran, Florian Weber, Grischa Weigand during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in SS 2020.

## Table of Contents

- [Requirements](#requirements)
- [Prerequisites](#prerequisites)
- [Pre-trained model](#pre-trained-model)
- [Running](#running)
- [Acknowledgments](#acknowledgments)

## Requirements

- Python 3.6
- OpenCV 4.0 (or above)
- Jetson TX2
- Jetpack 4.2
- asyncio (3.4.3)
- Cython (0.29.19)
- numpy (1.18.4)
- pip (9.0.1)
- pkg-resources (0.0.0)
- protobuf (3.12.2)
- PyYAML (5.3.1)
- setuptools (39.0.1)
- six (1.15.0)
- torch (1.4.0)
- tqdm (4.46.0)
- typing-extensions (3.7.4.2)
- utils (1.0.1)

## Prerequisites

1. Install pytorch

```
https://pytorch.org/
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Optional install CUDA

## Pre-trained models <a name="pre-trained-models"/>

lightweight-human-pose-estimation-3d-demo.pytorch
https://drive.google.com/file/d/1niBUbUecPhKt3GyeDNukobL4OQ3jqssH/view

MS-G3D - PyTorch implementation of "Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition", CVPR 2020
https://drive.google.com/file/d/1y3VbEnINtyriy82apiTZJtBV1a3cywa-/view

Pre-trained model is available at pretrained-models/

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):

```
python HumanActionRecognition.py  --video <webcam_id | filepath> -lhpe3d <pathToModel> -msg3d <pathToModel> --allcategories <y|n>
```

## Acknowledgments

This repo is based on

- [MS-G3D](https://github.com/kenziyuliu/MS-G3D)
- [lightweight-human-pose-estimation-3d](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch)

  Thanks to the original authors for their work!

## Contact

Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
