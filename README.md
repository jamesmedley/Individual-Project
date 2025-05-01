# Individual Project Code Repository

- Each experiment has its own branch
- Pytorch baseline U-Net implementation stems from https://github.com/milesial/Pytorch-UNet
- UNet++, ResUNet, ResUNet++, DUCK-Net Pytorch implementations are from https://github.com/zh320/medical-segmentation-pytorch
- Dockerfile and requirements.txt suitable for all experiments are detailed below.

## Dockerfile
``` Dockerfile
FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /workspace/project

# Clone the pytorch_wavelets repository and install it
RUN git clone https://github.com/fbcotter/pytorch_wavelets /workspace/pytorch_wavelets
RUN cd /workspace/pytorch_wavelets && pip install .

# install additional dependencies listed in requirements.txt
ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
```

## requirements.txt
``` txt
kymatio==0.3.0
matplotlib==3.6.2
numpy==1.23.5
opencv_python_headless==4.11.0.86
Pillow==9.3.0
pytorch_wavelets==1.3.0
tqdm==4.64.1
wandb==0.13.5
PyWavelets>=1.0.0
scikit-optimize
einops
```
