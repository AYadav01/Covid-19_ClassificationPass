# Covid-19_ClassificationPass
The repository contains deep learning (Densenet121, Resnet18) and ML based (Random Forest) aproach to analyse a collection of chest X-rays containing Normal, Pneumonia, and Covid-19 cases. The project consists of two parts:
* Normal vs. Abnormal Detection - In this case, the model tries to classify chest X-rays as either normal or abnormal (having pneumonia or covid).
* Covid vs. Pneumonia Detection - In this case, the model tries to classfy chest X-rays as either covid or pneumonia.

## Directory Structure
```
deepModels\ -> Contains 3 Deep learning models implementation
featureFiles\ -> Contains feature files required to run Random Forest classifier
model_checkPoints\ -> Contains saved weights for the trained CNNs
randomForest\ -> Contains Random Forest implementation
```
#### Running Random Forest
* The ```randomForest``` folder contains the data as well as the notebook, so the classifier can be executed directly without making modifications to any file path.

#### Running Densenet121 and Resent18
* The ```model_checkpoints``` folder contain the weights for the CNN models. The data should be loaded first as ```Dataloader``` and the weights should be loaded in the model second. The **prediction** section of the jupyter notebook should then be executed. 

### Prerequisites
Running **Jupyter Notebooks** requires correct path to the input data and the following **packages** for ```python-3.x```

```
numpy
sklearn
matplotlib
scikit-image
cv2
pytorch
SimpleITK
radiomics
tqdm
six
imblearn
```

### Data Statistics

* Number of normal cases = 7966
* Number of pneumonia cases = 8521
* Number of covid cases = 89

### Data Processing
* The chest X-rays are normalized to a mean of 0 and a variance of 1. The pixel values are scaled between 0 and 1.
* The number of covid cases are upsampled by 334 times -> 89 x 34 = 3026 variations.
* For the purpose of using CNNs with CUDA, the data was resized to a tensor size of [1, 256, 256].

### Feature Extraction
* The first step of the pipeline was to extract radiomics feature from the X-rays. In order to do so, mask for each samples are generated using a UNet model.
* Once masks are generated, it is further processed such that only masks with two contours in them that are of certain area (3654 px) are kept.
* The image with their respective mask are used to generate 4 class level radiomic features: **FirstOrder, GLCM, GLRLM, GLSZM, GLDM**.

## Models Used
For all the **models** mentioned below (**except UNET**), **4800** samples used for **training** and **1200** for **validation** (3000 samples per class was kept for a 50/50 balance).

**UNET**: This model is trained on **Montgomery dataset** to segment the lung from the entire chest X-ray. The Montgomery dataset contains images from the Department of Health and Human Services, Montgomery County, Maryland, USA. The dataset consists of 138 CXRs which are 12-bit gray-scale images of dimension 4020 × 4892 (which were resized to 256 x 256) . Only the two lung masks annotations are available which were combined to a single image in order to make it easy for the network to learn the task of segmentation. 

* ##### UNET Parameter
1. **Pixel-wise Cross Entropy** Loss is used with SGD at lr = 0.001.
2. 138 images were upsample to 1000 image for training.
2. The model is trained for 200 epochs.
3. 512 features for Bottleneck.
4. Lowest Validation loss acheived: 1.47.

**Densenet121**: This model is trained for both part of the project (Normal vs. Abnormal and Pneumonia vs. Covid)

* ##### Densenet Parameters
1. **Cross Entropy** Loss is used with **SGD** at a **lr = 0.001**.
2. The model is trained for **200 epochs**, however, early stopping is deployed if there is no progress happening in loss metric after 10-15 iterations.
3. For Normal vs. Abnormal task, the model stops at around epoch **114** (lowest validation loss: **1.21**, Validation Accuracy: **0.914**).
4. For Covid vs. Pneumonia task, the model stops at around epoch **150** (lowest validation loss: **0.228**, Validation Accuracy: **0.988**).

**Resnet18**: This model is trained only for Normal vs. Abnormal task to compare it with the Densenet121 model.

* ##### Resnet18 Parameters
1. **Cross Entropy** Loss is used with **SGD** at a **lr = 0.001**.
2. The model is trained for **200 epochs**, however, the model stops at around epoch **178** via early stopping (lowest validation loss: **1.440**, Validation Accuracy: **0.890**).

**Random Forest**: A non-deep learning based approach is also used to compare its performnace with state-of-the art CNNs.

* ##### Random Forest Parameter
1. Number of Trees = 5.
2. Maximum Depth = 10.
3. Number of Samples used = 6000.
4. Test Split = 0.2.
5. PCA_VARIANCE = 0.99.

### Results

##### For Abnormal vs. Normal Task (Precision, Recall, Auc)
Type | Desnet121 | Resnet18 | Random Forest
--- | --- | --- | --- |
Normal | (0.91, 0.92, 0.91) | (0.88, 0.91, 0.88) | (0.83, 0.86, 0.80) |
Abnormal | (0.92, 0.91, 0.91) | ( 0.91, 0.87, 0.88) | ( 0.79, 0.75, 0.80) |

#### For Covid vs. Pneumonia Task (Precision, Recall, Auc)
Type | Desnet121 | Random Forest
--- | --- | ---- |
Pneumonia | (1.0, 0.98, 0.98) | (0.99, 1.0, 0.99) |
Covid | (0.98, 1.0, 0.98) | (1.0, 0.99, 0.99) |

## Questions ?
* Which approach achieved overall better performance and why?
    - The Densenet121 as well as Random Forest seems to acheive similar results compared to Resenet18, however, Random Forest acheives the performance at a much faster pace. The reason that Densent121 might not be ideal in this case is because we dont have sufficient Covid data compared to other classes to train deep CNNs. Oversampling might introduce some form of
    redundancy in the data that could harm the model performance. 

* How did you handle the class imbalance for predicting pneumonia versus COVID-19?
    - For feeding to the CNNs models, Covid-19 dataset was augment approximately 34 times using various transformations (horizontal_fli, vertical_flip, rotations, shear operations etc.) to balance out the classes. For the radiomics features,
    Synthetic Minority Oversampling technique (computes k-nearest neigbor for a random selected minority point in dataset) is used to balance out the minority class.
    
* What further improvements would you make on Approach 2?
    - We can try using different combination of loss functions and optimizers to see how the model loss changes over time. Also, collecting better data and seeking out novel normalization techniques could significantly improve the model.

## Conclusion
* Even though the **Precision, Recall and Auc metrics** looks **promising**, the **model** still **needs** to be **validated** against a variety of **exteranal data sources** before any substanial **conclusions** are **drawn**.