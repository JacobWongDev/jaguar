# Jaguar Setup

Jaguar is meant to be run on linux. Below is a list of all the required software, and some imporant notes for using Jaguar on WSL.

To start, one should run

```./setup.sh```

since it will install vcpkg and download spdlog, both of which are Jaguar dependencies.

## Required Software

Below are the required softwares for building/developing Jaguar.

### CMake and make

To install the latest CMake on WSL, do:

1. Download CMAKE tarball from [here](https://cmake.org/download/)
2. ```tar -xvzf "cmake tarball name"```
3. ```./bootstrap```
4. ```make -j$(nproc)```
5. ```sudo make install```

To get make, you can use APT.

### CUDA Toolkit

The CUDA Toolkit comes with nvcc and a collection of useful libraries. It can be downloaded [here](https://developer.nvidia.com/cuda-downloads).

Follow the steps at the link.

As a final step, add the following to ~/.bashrc

```bash
export PATH=$PATH:/usr/local/cuda-<!version here!>/bin
```

### NVIDIA GPUs and WSL

To allow WSL to use the GPU, one must follow [this guide](https://canonical-ubuntu-wsl.readthedocs-hosted.com/en/latest/tutorials/gpu-cuda/).

## VSCode extensions

To develop CUDA Kernels, the "Nsight Visual Studio Code Edition" works well with the C/C++ extension pack. To ensure that include statements are resolved properly by these plugins, make sure to add a line to settings.json which will ensure both cuda toolkit and vcpkg library headers are not reported as errors.

Here is what my settings.json looks like:

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/cuda-12.5/include",
                "${workspaceFolder}/vcpkg/installed/x64-linux/include/"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c23",
            "cppStandard": "gnu++23",
            "intelliSenseMode": "linux-gcc-x64",
            "configurationProvider": "ms-vscode.makefile-tools"
        }
    ],
    "version": 4
}
```

## CUDA Compute Sanitizer

To use NVIDIA's compute sanitizer tool (very useful!) on WSL, you may have to follow the steps below.

The tool comes with the CUDA toolkit, however for some versions of the toolkit there is a known problem when used with WSL which you can read about [here](https://forums.developer.nvidia.com/t/compute-sanitizer-help-errors-on-wsl2-ubuntu-22-04/295507)

The solution is to run the following in Administrator powershell:

```ps
REG ADD HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm /f /v EnableDebugInterface /t REG_DWORD /d 1
REG ADD “HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger” /f /v EnableInterface /t REG_DWORD /d 1
```
