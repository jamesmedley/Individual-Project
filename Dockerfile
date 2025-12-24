FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /workspace/project

# Clone the pytorch_wavelets repository and install it
RUN git clone https://github.com/fbcotter/pytorch_wavelets /workspace/pytorch_wavelets
RUN cd /workspace/pytorch_wavelets && pip install .

# install additional dependencies listed in requirements.txt
ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt