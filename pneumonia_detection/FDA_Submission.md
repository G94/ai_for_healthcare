# FDA  Submission

**Your Name:**
Cesar Gustavo Seminario Calle

**Name of your Device:**
Identification of Pneumonia from X-Ray imaging

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** 

 For assisting the radiologist in the detection of Pneumonia using x-rays.


**Indications for Use:**

- Emergency workflow re-prioritization.
- This algorithm is intended for patients (women and men) between the ages of 20 to 60 years old whose X-ray has been taken in PA, AP position.
- The patients can have zero or more disease, specifically Atelectasis, Edema and Infiltration.

**Device Limitations:**
- 
- The algorithm will take more time at inference if it is not run in a GPU.
- Patients over 60 years old would cause the algorithm output not valid results.


**Clinical Impact of Performance:**
- A `false positive` will  overcharge the priority list, making the radiologist focus on cases which don't need immadiatly attention.
- A `false negative` will have a high impact in the prioritization of  the quee, putting patient in high risk in the final positions of the priority list.

### 2. Algorithm Design and Function


![flow_chart](img/flow_chart.PNG)


**DICOM Checking Steps:**
- The first validation is to check if the `patient position` is **PA** or **AP**.
- The second validation is to check the `modality` value is **DX** (Digital Radiography).
- The third validation is to check that the `body part examined` is the **chest part**.


**Preprocessing Steps:**

Standarization:
1. Divide into 255 to scale the image
2. Extract the mean of the image

Normalization:
3. Divide into the standard deviation

**CNN Architecture:**

```
Model: "sequential_1"

Layer (type)                 Output Shape              Param #   
=================================================================
model_1 (Model)              (None, 7, 7, 512)         14714688  
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               12845568  
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 257       
=================================================================
- Total params: 27,691,841
- Trainable params: 15,336,961
- Non-trainable params: 12,354,880
```

### 3. Algorithm Training
**Parameters:**

* Types of augmentation used during training

    - rescale = 1. / 255.0
    - horizontal_flip =  False
    - vertical_flip = False 
    - height_shift_range = 0.05
    - width_shift_range = 0.05
    - rotation_range = 3
    - shear_range = 0.9
    - zoom_range = 0.1


* Batch size

    16

* Optimizer learning rate
    
    0.001

* Layers of pre-existing architecture that were frozen
    
    The first 17 layers of the pretrained model were frozen.


* Layers of pre-existing architecture that were fine-tuned


![VGG16](img/vgg_architecture.PNG)


```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
=================================================================
```
    - Total params: 14,714,688
    - Trainable params: 14,714,688
    - Non-trainable params: 0


* Layers added to pre-existing architecture
```
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))    
    
    model.add(Dense(256,  activation = 'relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1,  activation = 'sigmoid'))
```


----

![Train/Validation Loss](img/loss_plot.PNG)

----

![Precision Recall plot](img/precision_recall_plot.PNG)

**Final Threshold and Explanation:**

| |face-detection-adas-binary-0001|head-pose-estimation-adas-0001|landmarks-regression-retail-0009|gaze-estimation-adas-0002|
|-|-|-|-|-|
|BIN|1.7 MB|3.7 MB|373 KB|3.6 MB|
|XML|114 KB|50 KB|42 KB|65 KB|
|PRECISION|FP32|FP16|FP16|FP16|

The Final threshold was 0.3125



### 4. Databases

**Description of Training Dataset:** 
![Training Images](img/training_data.PNG)


**Description of Validation Dataset:** 

![Validation Images](img/validation_data.PNG)

### 5. Ground Truth

The dataset was labeled using Natural Language Processing from the associated radiological reports. The advantage of this method is that we can label many images in a short period of time.
A representative sample of this dataset labeled can be contrasted against an specialist if neccesary.

3. What are the limitations of the method through which the dataset was created ?
The model has overall 10% error, the dataset might contain some erroneous labels.

### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

**Ground Truth Acquisition Methodology:**

**Algorithm Performance Standard:**
