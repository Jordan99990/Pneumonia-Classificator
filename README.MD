# Pneumonia Detection

## Overview
This project classifies images of chest X-rays into two categories: Normal and Pneumonia. 
It uses a Convolutional Neural Network (CNN) for image classification using Keras.

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images). 
It contains chest X-ray images categorized into two folders: `NORMAL` and `PNEUMONIA`. The `NORMAL` folder contains X-ray images of healthy patients, while the `PNEUMONIA` folder contains X-ray images of patients diagnosed with pneumonia.

![Normal](/img/4.png)

The above images shows a batch of 5 images that are `Normal`

![Pneumonia](/img/5.png)

The above images shows a batch of 5 images that are `Pneumonia`

## Convolutional Models 
### Custom CNN Model
#### Architecture
The custom CNN model consists of the following layers:
1. **Conv2D**: 32 filters, kernel size (3, 3), activation 'relu', input shape (224, 224, 3)
2. **MaxPooling2D**: pool size (2, 2)
3. **Conv2D**: 64 filters, kernel size (3, 3), activation 'relu'
4. **MaxPooling2D**: pool size (2, 2)
5. **Conv2D**: 128 filters, kernel size (3, 3), activation 'relu'
6. **MaxPooling2D**: pool size (2, 2)
7. **Flatten**
8. **Dense**: 512 units, activation 'relu'
9. **Dropout**: rate 0.5
10. **Dense**: 1 unit, activation 'sigmoid'

#### Training Process
1. **Data Preprocessing**: Images are resized to 224x224 pixels and normalized.
2. **Compilation**: The model is compiled with binary cross-entropy loss and the Adam optimizer.
3. **Training**: The model is trained on the training dataset with a validation split, using data augmentation techniques to improve generalization.

### Pretrained VGG16 Model
#### Architecture
The VGG16 model is a deep convolutional network pre-trained on the ImageNet dataset. For this project, the following modifications are made:
1. **VGG16 Base**: Pre-trained VGG16 model without the top layers.
2. **Flatten**
3. **Dense**: 256 units, activation 'relu'
4. **Dropout**: rate 0.5
5. **Dense**: 1 unit, activation 'sigmoid'

#### Training Process
1. **Data Preprocessing**: Images are resized to 224x224 pixels and normalized.
2. **Feature Extraction**: The pre-trained VGG16 model is used to extract features from the images.
3. **Fine-Tuning**: The custom top layers are trained on the extracted features, while the lower layers of VGG16 are optionally fine-tuned.
4. **Compilation**: The model is compiled with binary cross-entropy loss and the Adam optimizer.
5. **Training**: The model is trained on the training dataset with a validation split, using data augmentation techniques to improve generalization.


## UI
The user interface for this project is built using Streamlit. It allows users to upload chest X-ray images and get predictions from the trained models.

### Running the Streamlit App
To run the Streamlit app, use the following command:
```sh
streamlit run app.py
```

## Screenshots
![1](/img/1.png)
![2](/img/2.png)
![3](/img/3.png)