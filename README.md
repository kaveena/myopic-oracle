# Composition of Saliency Metrics for Channel Pruning with a Myopic Oracle

This code is used to run the experiments and recreate the graphs for the Composition of Saliency Metrics for Channel Pruning with a Myopic Oracle paper submitted to ICML 2020.

N.B. The AlexNet network is not supplied in this zip as it exceeded the file size limit for ICML supplementary material.  All the other networks used in the paper are provided. 

# Contents of the zip
  * Source code of modified Caffe with helpers for channel pruning
  * Python script to download and extract the CIFAR-10 dataset
  * Trained Caffemodels for LeNet-5, CIFAR10, NIN and ResNet-20 on the CIFAR-10 dataset
  * Source code for Composition of Saliency Metrics for Channel Pruning
  * Dockerfile to setup the environment

# Dependencies

The results presented in the paper were run on a machine with a single GTX 1080Ti

Requirements:
  * NVIDIA GPU with supported CUDA
  * Linux
  * Docker

  Docker needs to be installed see https://docs.docker.com/install/ for details on how to install Docker on your platform.  Your user needs to be added to the docker group to be able to run experiments without root permission.
  * NVIDIA Container Toolkit

  Install the NVIDIA Container Toolkit from https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support) and restart your docker service. To check that the GPUs can be access in Docker use:
  ``` 
  docker run --gpus all nvidia/cuda nvidia-smi
  ```

# Setup environment

The Dockerfile.gpu installs all the dependencies for our experiments in the Docker Container.  Build the Docker Container with from the root directory of the project:
```
docker build -t myopic-oracle -f Dockerfile.gpu .
```

Check that GPUs can be accessed using:
```
docker run --gpus all,capabilities=compute -it myopic-oracle nvidia-smi
```
If there is a driver mismatch error `Failed to initialize NVML: Driver/library version mismatch`, then you need to uncomment the lines in Dockerfile.gpu that installs a specific version of NVIDIA driver.  Replace the driver version in the format XXX.YY with the NVIDIA driver version found on the host machine. Find the driver version on the host machine using:
```
cat /proc/driver/nvidia/version
```
Rerun the build process. An error in linux kernel may occur. Change the Linux kernel version to the one from the error message.

# Run experiment

The experiments can be started with 
```
docker run --gpus all,capabilities=compute --mount source=$(realpath results),target=/data/,type=bind -it myopic-oracle ./run.sh
```

The experiments can be stopped and restarted at any time with the same command.

# Results
The results of the experiment are generated in results/Graphs.  On a single GTX 1080Ti the experiments for LeNet-5 took about 7min, CIFAR10 about 36min and 18h for ResNet-20.
