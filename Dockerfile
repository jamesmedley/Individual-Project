# Use a base image with PyTorch support
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Set the working directory inside the container
WORKDIR /workspace/project

# Clone the pytorch_wavelets repository and install it
RUN git clone https://github.com/fbcotter/pytorch_wavelets /workspace/pytorch_wavelets
RUN cd /workspace/pytorch_wavelets && pip install .

# Install dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
