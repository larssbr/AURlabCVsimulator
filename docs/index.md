# Welcome to AURlabCVsimulator
This repository is made for testing different computer vision methods by simulating a mission done in Trondheimsfjorden in April 2016

#### Libarys used

- Python: 2.7.11 —Anaconda 2.5.0 (64-bit)

- scipy: 0.17.0

- numpy: 1.10.4

- matplotlib: 1.5.1

- pandas: 0.17.1

- sklearn: 0.17

### Folders of images to run simulations on

- images_L&R_avoided obstacle first try
- repeatExperiment
- images close to transponder

#### Mehtods in AURlabCVsimulator

Stereo camera method:
- Dipsarity method

Mono camera methods also called the texture based methods:

- LBP ROI method
- Haralick ROI method

- SLIC Superpixel Locally Binary Pattern method
- SLIC Superpixel Harlick method

Bellow are some examples:

### Typical underwater image with an obstacle

Image with an obstacle

![imageTest](notebooks/LBPs/docsIMG/imageTest.png)

### EXAMPLE  OF THE LBP ROI method
Predicted image with lbp

![image_prediction_lbp](notebooks/LBPs/docsIMG/image_prediction_lbp.png)

Display maskedImage image

![maskedImage](notebooks/LBPs/docsIMG/maskedImage.png)

Display countour center and boundingbox of the predicted obstacle

![drawnImage_boundingBox_maskedImage.png](notebooks/countours/docsIMG/drawnImage_boundingBox_maskedImage.png)

### EXAMPLE  OF THE PREDICTION WITH Haralick ROI method

![image_prediction_lbp](notebooks/Haralick/docsIMG/image_predicted.png)

### EXAMPLE  OF THE PREDICTION WITH SLIC Superpixel Locally Binary Pattern method
![LBP_prediction_dots.png](notebooks/LBPs/LBP_prediction_dots.png)

### Stereo camera methods
- 
### EXAMPLE  OF THE Obstacle avoidance with Dipsarity method
![disparityImageClean](notebooks/disparity/disparityImageClean.jpg)


### Usefull links to understnad parts of the code faster

Unifrom Local Binary Pattern (watch this to understand better): 
- https://www.youtube.com/watch?annotation_id=annotation_98709127&feature=iv&src_vid=wpAwdsubl1w&v=v-gkPTvdgYo

####
- Lars Brusletto Master thesis


