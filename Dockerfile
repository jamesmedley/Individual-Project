# Use a base image with PyTorch support
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Set the working directory to /workspace/unet
WORKDIR /workspace/unet

# Clone the pytorch_wavelets repository and install it
RUN git clone https://github.com/fbcotter/pytorch_wavelets /workspace/pytorch_wavelets
RUN cd /workspace/pytorch_wavelets && pip install .

# Install any additional dependencies listed in requirements.txt
ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt

# Add the rest of the code to the container
ADD . .

# Set the default command to run the training script with parameters
CMD ["python3", "train.py", "--epochs", "50", "--batch-size", "8", "--learning-rate", "0.001", "--classes", "1"]
