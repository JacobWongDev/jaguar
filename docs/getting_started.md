Installations on WSL
$ sudo apt-get update
$ sudo apt install g++
$ sudo apt install make
$ sudo apt-get install libssl-dev

Download CMAKE tar.gz from https://cmake.org/download/
- Install CMAKE:
$ tar -xvzf <cmake tarball>
$ ./bootstrap
$ make -j$(nproc)
$ sudo make install

Download CUDA Toolkit
- https://developer.nvidia.com/cuda-downloads


Enable GPU usage on WSL2
(Full guide here, but by now some steps are already done)
https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview

1. Go here to download correct driver https://www.nvidia.com/Download/index.aspx?lang=en-us

add the following to ~/.bashrc
$ export PATH=$PATH:/usr/local/cuda-12.5/bin

To enable the compute-sanitizer for memcheck and other utils, execute::

REG ADD HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm /f /v EnableDebugInterface /t REG_DWORD /d 1
REG ADD “HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger” /f /v EnableInterface /t REG_DWORD /d 1

Source: https://forums.developer.nvidia.com/t/compute-sanitizer-help-errors-on-wsl2-ubuntu-22-04/295507

For VSCode users 

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