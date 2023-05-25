# Color-Skepticism
A computer vision course project based on SOTA colorization techniques

### Objective
- The primary objective of the project is to analyze and assess different image colorization models by subjecting the input images to several manipulations using computer vision techniques.
- The image manipulations can be simple disturbances like blurring, rotating, flipping, activating random pixels to more complicated disturbances such as derivatives, alpha blending , object replacement, neural style transfer.
- The project attempts to analyze the robustness of image colorization models and comment on their failcases.

### Data
We compiled a collection of licensed Kaggle datasets. In the future, for more complex models, we will be using bigger datasets such as ImageNet, COCO.

### Existing Colorization models
We have collected 5 working image colorization models. We have integrated 2 of these models for the sample run.

###  Image disturbances/manipulations
We introduced basic image manipulations like Blurring,Rotating, Flipping in place to images taken from the dataset. 

### Workflow
- All the image manipulations and output similarities and scores are stored in a csv file. After thousands of iterations, this csv file will have enough correlations to run machine learning models that can identify model behaviors. 
- Non-convex optimization can be applied to find global maxima. These global maxima will correspond to the image disturbances that confuse the models the most. Hence, these are failcases.
- However, better representation of model evaluation might be required for accurate results.

### Evaluation
We have developed code to calculate FID scores and Image similarity scores based on the python package SEWAR. Better evaluation might be required for increased model performance evaluation.

### Challenges
- Finding Pretrained models for Image Colorization.
- An ongoing challenge for the project is to figure out the right evaluation metric. The code implementation will identify and return fail cases depending on the values returned by the evaluation function. , we are using FID and SEWAR similarity scores which may not give ideal results for improving model performance.
- Human Evaluation may be necessary. Before subjecting to evaluation, intermediate steps such as
KNN might be useful to look into.

### Future Work
- Looking into more complicated image disturbances like edge manipulations , texture transfer, object detection and manipulation.
- Finalizing our evaluation metrics for the model performances.
- The CSV file with the performance results can be analyzed using the Machine Learning models and optimization techniques to better understand the fail cases.


### Packages and Environment
- Packages/Libraries:
    - OpenCV
    - Pillow
    - Numpy
    - Pandas
    - Matplotlib
    - Sewar
- Pretarined Models / Neural Networks
- Environment:Anaconda Environment

### References
- Input Processing
https://docs.opencv.org/4.x/
https://medium.com/featurepreneur/blending-images-using-opencv-bfc9ab3697b7
- Colorization Models:
    - https://github.com/richzhang/colorization
    - https://github.com/lukemelas/Automatic-Image-Colorization/
    - https://github.com/moein-shariatnia/Deep-Learning/tree/main/Image%20Colorization%20Tutorial
- Evaluation Metrics:
    - https://sewar.readthedocs.io/en/latest/
    - https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
