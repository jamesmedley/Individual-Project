# Third-year Research Project
This repository contains the software associated with my third-year undergraduate individual research project and
dissertation, titled **_Exploring the Integration of Wavelet Transforms with U-Net for Polyp Segmentation_**, which was
awarded 84%.
___ 

## Research Overview
This project explores the integration of wavelet transforms into the U-Net architecture to improve semantic segmentation
in medical imaging, with a focus on polyp segmentation. Two complementary approaches are investigated: replacing parts
of the U-Net encoder with a fixed wavelet scattering network, and substituting standard downsampling and upsampling
operations with real- and complex-valued wavelet transforms.

A series of hybrid scattering models (WSN-UNet-J1–J4) are proposed and evaluated to assess how the depth of encoder
replacement affects segmentation performance and data efficiency, particularly in low-data regimes. Wavelet-enhanced
U-Nets (LSW-iDWT and LSW-iDTCWT) are developed to improve the reconstruction of fine spatial
detail through wavelet-based upsampling in the decoder.

This work demonstrates that integrating wavelet-based signal processing techniques into U-Net can provide both
performance and interpretability benefits for medical image segmentation. While hybrid scattering encoders show improved
data efficiency and faster convergence in low-data regimes, the most consistent gains are achieved by replacing
conventional pooling and upsampling with wavelet transforms. In particular, the DTCWT-based model preserves fine spatial
detail through its directional selectivity and approximate translation invariance, outperforming strong U-Net-based
baselines and generalising well across datasets. Overall, these results highlight the potential of wavelet-enhanced
architectures as a principled and effective alternative to standard CNN design choices in medical segmentation tasks.

The full dissertation is provided [here](docs/dissertation.pdf) for reference.

___

## License
- The dissertation PDF remains copyright James Medley 2025. All rights reserved; no part may be reproduced, quoted, or distributed without prior written consent of the author.  
- Code is licensed under [MIT](LICENSE).

___

## Setup

### Docker
The following [Dockerfile](Dockerfile) is suitable for all experiments:
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

### Python environment
Required Python packages are listed in the [requirements.txt](requirements.txt):
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

___

## Experiments

___

## Models

___

## Results
The directory [results-and-checkpoints](results-and-checkpoints) contains raw results, charts, learned filter
visualisations, layer activation maps, and sample segmentation masks for each evaluated model. There is also a
_checkpoint.pth_ file provided for each model.

___

## References
- Pytorch baseline U-Net: https://github.com/milesial/Pytorch-UNet  
- UNet++, ResUNet, ResUNet++, DUCK-Net: https://github.com/zh320/medical-segmentation-pytorch
