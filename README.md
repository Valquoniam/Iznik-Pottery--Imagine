
# TileGAN : Generating Iznik Tiles visual dictionary

- ## How to use the code

You'll absolutely need to use a conda environment to use the codebase.
I recommand to use miniconda.
You can download it [here](https://docs.conda.io/en/latest/miniconda.html), on both Windows & Linux. 

Don't forget to reload your terminal for conda to initialize.
Once in the (base) environment in conda, setup libmamba for faster installation :

```
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Now you can create the environment :

- Linux users : `conda env create -f configs/tilegan_linux.yml`
- Windows users : `conda env create -f configs/tilegan_windows.yml`

The installation is still a bit slow (~5 to 10 mins)

To activate the environment, you can now type `conda activate tilegan`

# System requirements 

The code needs cuda runtime version 11.1 to run the code. You can download it [here](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Windows&target_arch=x86_64) for both Windows & Linux. Also, you may have to run `sudo apt install cuda-toolkit` for nvcc --version to work.


## Linux (or WSL) : Download the runtime (local) version. 

- To setup it, after the installation, run this:
    ```
    export CUDA_HOME="/usr/local/cuda-11.1"
    export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-11.1/bin:$PATH"
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-11.1 /usr/local/cuda

    ```	

    Then run nvcc-version to check if it's working.
    Here is the expected terminal output :

    ```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    ...
    Build cuda_11.1.TC455_06.29069683_0
    ```
- Last minor thing : gcc < 10 is required, 
    Check your version with `gcc --version`

    If it's too high, here how to downgrade it :
    ```
    sudo apt install gcc-9 g++-9
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 10
    
    sudo rm -rf /usr/bin/gcc
    sudo rm -rf /usr/bin/g++

    sudo ln -s /usr/bin/gcc-9 /usr/bin/gcc
    sudo ln-s /usr/bin/g++-9 /usr/bin/g++
    ```

    Then check again with `gcc --version`. Here is the expected output :
        
        ```bash
        g++ (Ubuntu 9.5.0-1ubuntu1~22.04) 9.5.0
        Copyright (C) 2019 Free Software Foundation, Inc.
        This is free software; see the source for copying conditions.  There is NO
        warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
        ```

If, at any time, the code stops working, like after a reboot or something, check `nvcc --version` and `gcc --version` to see if they are still working. If not, run the commands above again. You can also place them in the .bashrc file to run them automatically at startup.	No need for the sudo commands in the .bashrc file.

## The code itself

Once everything is setup, launch `python main.py` to download the pretrained TileGAN model.

Functionalities :

- Generate images : `python main.py -g`. By default, 32 images are stored in _results/images_.
    - You can select the number of images you want to generate with `-i`. Ex : `python main.py -g -i 50`
- Download the dataset I built : `python main.py -d`. It will be stored in _dataset/iznik_.
- Generate latent vectors : `python main.py -gl`. By default, 10 are created. For that, you need to download the dataset.
- Open the website I made to display my results : `python main.py -w`.
    - The website has plenty of functionalities : observe the generated images, generate 32 new ones, save them as a grid, ...
- Open the tensorboard page to see how the training went : `python main.py -tb`
    - To open the projector page, to evaluate the diversity of the generated images : `python main.py -tb -pr`

Of course, you can have these arguments in a row. Example : To open the website & the tensorboard page while generating 200 images with a diversity rate of 1.0 :

`python main.py -g -i 200 -t 1.0 -w -tb`
