# Traffic Analysis Project, Indianapolis Tactical Urbanism

This project analyzes traffic data to provide insights and visualizations.

## Prerequisites

**Conda Installation:**  
     
   * Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended, lighter) or [Anaconda](https://www.anaconda.com/products/distribution) (full distribution with many pre-installed packages). Follow the installation instructions for your OS.
   * Once installed, open your terminal/command prompt and verify `conda` is working:  
       
     conda --version

     
**CUDA Toolkit & cuDNN (if using CUDA/Nvidia GPU):**  
     
   * This is crucial for GPU acceleration and *must be installed on your system outside of conda*.  
   * **Download CUDA Toolkit 11.8 or 12.x** from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads). Ensure it's compatible with your GPU.  
   * **Download compatible cuDNN** from [NVIDIA's cuDNN page](https://developer.nvidia.com/cudnn).  
   * Follow NVIDIA's instructions for installation. Ensure the CUDA toolkit's `bin` directory is added to your system's `PATH` environment variable.

## 1. Create and Activate a Conda Environment

a. Create a new conda environment and give it a name (I used 'yolov11_env') with Python 3.10

`conda create -n yolov11_env python=3.10`

b. Activate the newly created environment

conda activate yolov11_env

*(You'll see `(yolov11_env)` prepended to your command prompt, indicating the environment is active.)*

## 2. Install Core Packages 

Now, we'll install PyTorch with CUDA support, Ultralytics (which includes YOLOv11), and Jupyter/IPython kernel into our active `yolov11_env`.

**Important Note for PyTorch:** While `conda install pytorch` is common, for specific CUDA versions like `cu128`, using `pip` with the `--index-url` is often the most reliable way to ensure you get the exact pre-built wheels. `pip` works perfectly within conda environments.

a. Ensure pip is up-to-date within the environment

pip install --upgrade pip

b. Install PyTorch with CUDA 128 support

> This must be done AFTER activating the conda environment

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

c. Install Ultralytics and other necessary packages

`pip3 install ultralytics opencv-python pillow matplotlib numpy jupyter ipykernel`

*(This step might take a few minutes as it downloads all the packages.)*

3. Launch Jupyter Notebook

With our environment set up, we can now launch Jupyter Notebook.

a. Make sure your 'yolov11_env' is active

`conda activate yolov11_env`

b. Launch Jupyter Notebook from this project's root folder

`jupyter notebook`

Your web browser should open to the Jupyter interface.

Open the jupyter notebook, read the instructions, and run the blocks.

## License
MIT License