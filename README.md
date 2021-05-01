# yolov4-custom-functions
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This is a tensorflow YOLO-V4 implementation of knee joint detection and automatic cropping. 

## Getting Started
### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Using Custom YOLOv4 Weights on Detecting Knee Joint

  * Copy and paste your [**custom.weights**](https://drive.google.com/file/d/1VdA1BS4oJUX9-oQ_qviK_5KKtvMclYJZ/view?usp=sharing) file into the 'data' folder.
  * Build the model with the following command:
  ```bash
  python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 
  ```
  * Upload the images to **data/images/** folder.
  * Start cropping the image with the following command (you can remove **--dont_show** if you want to see the images with bounding box pop out in your screen):
  ```bash
  python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/ --crop --dont_show
  ```
  * The cropped images will go to the **detections/crop/** folder

### References  

  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions)
