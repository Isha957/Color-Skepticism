# Color-Skepticism
A computer vision course project based on SOTA colorization techniques

### Objective
- The primary objective of the project is to analyze and assess different image colorization models by subjecting the input images to several manipulations using computer vision techniques.
- The image manipulations can be simple disturbances like blurring, rotating, flipping, activating random pixels to more complicated disturbances such as derivatives, alpha blending , object replacement, neural style transfer.
- The project attempts to analyze the robustness of image colorization models and comment on their failcases.

## Existing Work
Over the past few years, deep learning models have made significant advances in predicting the most conceivable colors to objects. The earliest deep learning networks incorporated simple CNN layers stacked linearly with a RelU and Batch-normalization layer. The input image was of 256 x 256 and the model resulted in a 224 x 224. This model used class rebalancing technique and was implemented using the Caffe framework. User guided frameworks demand human intervention through input in the form lines or scribbles. 

- ***Zhang et al.*** - proposed a fully automatic approach that implements CNN structure with 10 blocks , each block containing 2 or 3 conv and ReLu layers followed by a BatchNorm layer. The CNN is trained to map the grayscale image over quantized color value outputs. This model is claimed to have fooled humans on 32% of the test cases.

- ***PaletteNet*** - A deep learning architecture that takes two inputs - source image(image that is to be recolored) and a target palette. The model outcome is an image recolored with respect to the target palette. [Junhocho et al.] proposed an encoder-decoder framework (feature encoder and recoloring decoder with Euclidean/Adversarial loss functions. 

- ***Deoldify*** - A CNN based colorization technique primarily aiming to restore color information to old and historic images.The Generator applies colors to the objects that it recognizes being trained on and the discriminator criticizes the choice of the colors applied.The colorizing network is a U-Net constructed using a pretrained ResNet model. The ResNet is for the encoding and a decoding to reconstruct the color image from the representation derived by the ResNet. 

- ***Deep AI ParaColorizer*** - Another automatic colorization framework that uses two parallel frameworks to color the foreground and background of the image separately. First , foreground GAN model training to detect objects followed by background model training on real-world examples and append to the main dataset. Third, a fusion deep learning model trains on the main dataset and combines the outputs of both foreground and background GAN models.

> Please refer to the project report uploaded in the repository for further understanding on the working/ implementation of these model architectures.

## Workflow
A proper design of the model is essential to analyze the Colorization models effectively. For the purpose of this project, we identified 4 working parts: 

- *Disturbances:* The part of the code that applies randomized disturbances to input images.
- *Colorization models*: This part keeps track of all the available colorization models, APIs etc. When provided with a gray image, this class returns a list of colorized images for different models.
- *Evaluation*: Part 3 of the code - Responsible for identifying the differences between input image and recolored image. 
- *Recording*: All the disturbances, performance metrics are recorded, the images are saved with globally recognizable IDs.
Each part works relatively independently and the main part of the code connects all the parts together. The code also provides a command line user interface for ease of access.

*The main challenge of this project was to place all of the individual blocks together into a working pipeline. Because the project has a lot of moving parts, it was prone to complicated bugs that resulted due to errors in multiple parts of the code. This challenge was overcome by developing the project with maximum amount of relative independence, such the dynamic errors in the code would not influence each other.*

> Our framework comprises four significant modules that carry a raw RGB image to the output recolored image and end with recording the similarity scores pertaining to each model.

### Data
For the scope of this project, we plan to utilize the COCO dataset which is rich with images of vivid categories (328,000 images). 

###  Image manipulations
After initial preprocessing, the image is sent to methods that impart disturbances. The goal of the disturbances is to randomly disturb the content of the image that is independent from color - such as brightness, contrast, picture orientation, objects, edges etc. Disturbances that we applied are varied significantly -from simple disturbances such as  blurring, noise(additive and multiplicative), flipping, rotation, invert, Sharpening, Histogram Equalization (increasing contrast) etc to advanced disturbances such as Alpha Blending, Neural Style Transfer, modified Convolution filtering. 

### Disturbances
- *Convolution*
    1. Edge Manipulation 
    2. Random Convolve

- *Neural Style Transfer*

To add a textural disturbance to the image that puts the colorization models to test, we collected available dataset of texture samples(referenced). Texture Transfer  manipulates edges of the images but preserves the Gestalt principle of the image. Our objective behind using style transfer is to understand how of much of this gestalt information is accurately grasped by the colorization models.

- *Alpha Blending*
-
Alpha Blending refers to the process of overlaying one image over the another in such a way that the second image becomes background with translucency. With the  control of the alpha parameter, We are using alpha blending to test the recolorization model’s choice of when the image boundaries become less obvious. 

- The images with added disturbances are converted to greyscale and input to the colorization models.

> For additional understanding of disturbance mechanisms and implementation refer to our report uploaded in the repository.

### Colorization models
In this project, we only consider some of the popular colorization models such as the ECCV16 and SIGGRAPH17 proposed by (Zhang’s paper citation), Palette by (Junho Cho et al.), Deoldify by (Jason Antic and Fast AI), DeepAI’s Paracolorizer models. Each of these models is fed the greyscale form of the disturbed RGB image. The RGB output of these models are stored and sent to similarity score methods that compare the output against the original.

- All the image manipulations and output similarities and scores are stored in a csv file. After thousands of iterations, this csv file will have enough correlations to run machine learning models that can identify model behaviors. 
- Non-convex optimization can be applied to find global maxima. These global maxima will correspond to the image disturbances that confuse the models the most. Hence, these are failcases.
- However, better representation of model evaluation might be required for accurate results.

### Evaluation Metrics
Perfect evaluation of colorization models requires human analysis. However, performance metrics such as MSE are really effective in identifying the quality of colorization when applied carefully.  
A couple of Performance metrics are used - This is because colorization behaves very differently based on the use case. So, it would be wise to consider the colorization quality from multiple angles.
- Two approaches of comparison:
1.  comparing the recolored image against plain input RGB image - led to erroneous similarity scores. This was because the recolored image retains the disturbances added and comparing against original RGB with no disturbances would penalize the scores more for the disturbances over the color choice . Hence the outcome was an inaccurate representation of the colorization model performance. 
2.  Our final evaluation approach is comparing the recolored output against a disturbed RGB image. It was important to not lose the color information while applying the disturbances. Then the disturbed RGB image is sent for evaluation, where the recolored image is only compared against the disturbed RGB image. This would only penalize scores for the faults in colorization which gives us more accurate feedback
- We have developed code to calculate FID scores and Image similarity scores based on the python package SEWAR. Better evaluation might be required for increased model performance evaluation.

## Observations
We gauge the performance of the model using the following metrics: 
- ***Histogram comparison:*** This method compares the histograms between the input and the recolored images. This has been proved to be the most effective, as many colorization models fail to color the image with as much variance in the color as the original image. 
- ***Mean Squared Error(MSE) for RGB and HSV spaces:*** Mean squared error on RGB space is a quick way to identify the difference between 2 images. However, it doesn’t accurately comment on the performance of the colorization models, as it unfairly punishes the model for choosing a different color. We might be interested in this behavior when we want to enforce the usage of a color (For instance, the sky should be colored blue instead of red). However, this doesn’t reflect the idea entirely. To alleviate this, we use MSE in HSV space, by dropping the Hue channel. This gives a much better understanding of error in color saturation.
 We store the similarity scores , the combination of disturbances applied and the RGB image id in a CSV file which will be used later on for analysis. 
 
## Results
### Best Cases



### Fail Cases

## Challenges
- Finding Pretrained models for Image Colorization.
- An ongoing challenge for the project is to figure out the right evaluation metric. The code implementation will identify and return fail cases depending on the values returned by the evaluation function. , we are using FID and SEWAR similarity scores which may not give ideal results for improving model performance.
- Human Evaluation may be necessary. Before subjecting to evaluation, intermediate steps such as
KNN might be useful to look into.

## Future Work
- Looking into more complicated image disturbances like edge manipulations , texture transfer, object detection and manipulation.
- Finalizing our evaluation metrics for the model performances.
- The CSV file with the performance results can be analyzed using the Machine Learning models and optimization techniques to better understand the fail cases.


## Packages and Environment
- Packages/Libraries:
    - OpenCV
    - Pillow
    - Numpy
    - Pandas
    - Matplotlib
    - Sewar
- Pretarined Models / Neural Networks
- Environment:Anaconda Environment

## References
- Input Processing
    - https://docs.opencv.org/4.x/
    - https://medium.com/featurepreneur/blending-images-using-opencv-bfc9ab3697b7
- Colorization Models:
    - https://github.com/richzhang/colorization
    - https://github.com/lukemelas/Automatic-Image-Colorization/
    - https://github.com/moein-shariatnia/Deep-Learning/tree/main/Image%20Colorization%20Tutorial
- Evaluation Metrics:
    - https://sewar.readthedocs.io/en/latest/
    - https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
