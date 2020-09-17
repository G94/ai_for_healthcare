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

### 2. Algorithm Design and Function

<< Insert Algorithm Flowchart >>

**DICOM Checking Steps:**



**Preprocessing Steps:**


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
    None

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
 (For the below, include visualizations as they are useful and relevant)

**Description of Training Dataset:** 


**Description of Validation Dataset:** 


### 5. Ground Truth



### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

**Ground Truth Acquisition Methodology:**

**Algorithm Performance Standard:**
