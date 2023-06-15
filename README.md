# Calis ML
## Table of Contents
- [Calis ML](#calis-ml)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Handwriting Classification](#handwriting-classification)
    - [Voice Classification](#voice-classification)
  - [File Descriptions](#file-descriptions)
  - [Library Versions](#library-versions)
  - [Dataset](#dataset)
    - [Handwriting Classification](#handwriting-classification-1)
    - [Voice Classification](#voice-classification-1)
  - [Model Selection](#model-selection)
    - [Handwriting Classification Model](#handwriting-classification-model)
    - [Voice Classification Model](#voice-classification-model)
  - [Tutorial: Running the Notebook in Google Colab](#tutorial-running-the-notebook-in-google-colab)
    - [Prerequisites](#prerequisites)
    - [Step 1: Opening the Notebook in Google Colab](#step-1-opening-the-notebook-in-google-colab)
    - [Step 2: Running the Notebook](#step-2-running-the-notebook)
    - [Step 3: Accessing the TFLite File](#step-3-accessing-the-tflite-file)
  - [Model Performance](#model-performance)
    - [Handwriting Classification](#handwriting-classification-2)
      - [Lowercase Alphabets](#lowercase-alphabets)
      - [Uppercase Alphabets](#uppercase-alphabets)
      - [Digits](#digits)
    - [Voice Classification](#voice-classification-2)
  - [References](#references)

## Introduction

This project focuses on developing machine learning models to classify voice recordings and handwritten characters for the Calis application, with the goal of enhancing children's literacy and numeracy skills. The project entails two specific classification tasks:
### Handwriting Classification

The Handwriting Classification task aims to achieve accurate classification of handwritten characters into a total of 62 classes. These classes encompass lowercase alphabets (a-z), uppercase alphabets (A-Z), and digits (0-9). To accomplish this task, a Convolutional Neural Network (CNN) model, a powerful deep learning architecture specifically designed for image classification, is employed. The CNN model excels at learning intricate patterns and features within the handwritten characters, enabling precise classification.
### Voice Classification
The Voice Classification task aims to categorize voice recordings into a total of 18 classes. These classes encompass a diverse range of utterances, including alphabets, words, and sentences. The primary objective is to accurately classify and identify the spoken content in the recordings. To accomplish this task, we utilize transfer learning by adapting the pre-trained YAMNet model, which is renowned for its effectiveness in sound classification tasks.
## File Descriptions

- [handwritten-classification](https://github.com/Calis-Top-10/Calis_ML/tree/main/handwritten-classification) folder contains two sub-folders:

   - **notebook**: This sub-folder contains a collection of Jupyter Notebook files that guide through the entire workflow of training and evaluating the handwriting classification model. Each notebook is meticulously crafted to provide a step-by-step approach, covering essential stages such as data preprocessing, feature engineering, model architecture design, hyperparameter tuning, and performance evaluation.
     - [handwritten_character_recognition_SMALL.ipynb](https://github.com/Calis-Top-10/Calis_ML/blob/main/handwritten-classification/notebook/handwritten_character_recognition_SMALL.ipynb): This notebook focuses on the task of handwritten character recognition, specifically for a smaller alphabets dataset.
     - [handwritten_character_recognition_capital.ipynb](https://github.com/Calis-Top-10/Calis_ML/blob/main/handwritten-classification/notebook/handwritten_character_recognition_capital.ipynb): : This notebook focuses on the task of handwritten character recognition, specifically for a capital alphabets dataset.
     - [handwritten_digit_recognition.ipynb](https://github.com/Calis-Top-10/Calis_ML/blob/main/handwritten-classification/notebook/handwritten_digit_recognition.ipynb): This notebook focuses on the task of handwritten character recognition, specifically for a digits dataset.

   - **tflite**: This sub-folder contains the TensorFlow Lite (TFLite) file generated after training the handwriting classification model.

- [voice-classification](https://github.com/Calis-Top-10/Calis_ML/tree/main/voice-classification) folder contains two sub-folders:

   - **notebook**: This sub-folder contains a Jupyter Notebook file that demonstrates the process of transfer learning using the pre-trained YAMNet model for voice classification. The notebooks walk through the steps of loading the pre-trained YAMNet model, adapting it to specific voice classification tasks, fine-tuning the model on the dataset, and evaluating its performance.
     - [calis_voice_classification.ipynb](https://github.com/Calis-Top-10/Calis_ML/blob/main/voice-classification/notebook/calis_voice_classification.ipynb): This notebook focuses on the task of voice classification
   
   - **tflite**: This sub-folder contains the TFLite file generated after training the voice classification model. 
## Library Versions
Here are the versions of the libraries used in this project:

- Librosa version: 0.10.0.post2
- Numpy version: 1.22.4
- Matplotlib version: 3.7.1
- Soundfile version: 0.12.1
- Pandas version: 1.5.3
- TensorFlow version: 2.12.0
- TensorFlow IO version: 0.32.0
- TensorFlow Hub version: 0.13.0
- Scikit-learn version: 1.2.2
- Scipy version: 1.10.1
- Pydub version: 0.25.1
- Cv2 version: 4.7.0
## Dataset
### Handwriting Classification
The dataset used for the handwriting classification model consists of three separate datasets, each containing 62 classes. These classes include lowercase alphabets, uppercase alphabets, and digits. Each sample in the dataset is represented by a PNG file, with a standardized size of 300x300 pixels. The images are in grayscale mode, which means they contain only shades of gray ranging from black to white.
- **Labels**: 
  - **Lowercase Alphabets**: Handwritten characters representing the lowercase letters of the alphabet (a-z).
  - **Uppercase Alphabets**: Handwritten characters representing the uppercase letters of the alphabet (A-Z).
  - **Digits**: Handwritten characters representing numerical digits (0-9).
- **Number of Samples**: 
  - **Dataset 1**: The first dataset, obtained from Kaggle, consists of a total of 798,402 PNG files of handwritten characters. The original dataset had white text on a black background. However, we have inverted the pixel values of the images to match the input format from Android, which is handwritten characters with black color and a white background. This inversion ensures consistency between the dataset and the input format for better compatibility and accuracy in the handwriting classification model.
  - **Dataset 2**: The second dataset, created by our team, comprises 512 PNG files of handwritten characters. This dataset was used for training and validation purposes.
  - **Dataset 3**: The third dataset, also created by our team, consists of 1,233 PNG files of handwritten characters. This dataset was used for fine-tuning our model.

- **Data Preprocessing**:
  - **Inversion of Pixel Values**: Before normalizing the pixel values, the handwritten images from the Kaggle dataset were inverted. 
  - **Normalization of Pixel Values**: After inverting the pixel values, the images were further processed by normalizing the pixel values to a specific range, typically between 0 and 1. 
  - **Removal of Corrupt Data**: To ensure the integrity and quality of the datasets, any corrupt or damaged data instances were carefully filtered out. 

The datasets can be accessed using the following links:

- [Kaggle Dataset](https://www.kaggle.com/datasets/sankalpsrivastava26/capital-alphabets-28x28)
- [Dataset 1](https://drive.google.com/uc?id=1EOnvYEpS7vGPaj196x7ozxP41mSJ-ZJk)
- [Dataset 2](https://drive.google.com/uc?id=1XsqlO58Wk5VbxSDC1LSgsaLVI0dWxZh9)
- [Dataset 3](https://drive.google.com/uc?id=1iwCF7vIpiOj75IhnZzszpNZb7AoewYtB)
### Voice Classification
The dataset utilized for this sound classification model consists of personally recorded human voice recordings. Each recording has an average duration of 4 seconds and is stored in the WAV file format with a sampling rate of 22.05 kHz. The dataset encompasses a diverse range of utterances, including alphabets, words, and sentences, spoken by multiple speakers. Each audio file is meticulously labeled with its corresponding class, facilitating supervised learning for the sound classification task.
- **Labels**: 
  - **Alphabets**: Voice recordings of individual letters from the alphabet.
  - **Words**: Voice recordings of various words spoken by different speakers.
  - **Sentences**: Voice recordings of complete sentences spoken by different speakers.

- **Number of Samples**: The dataset includes a total of 1,064 human voice recordings.
- **Data Preprocessing**:
  - **Augmentation using Pitch Shift**: In the voice classification task, we performed data preprocessing to augment the training dataset. One of the augmentation techniques employed is pitch shifting. Pitch shifting allows us to modify the pitch of the audio, simulating different voice variations and increasing the diversity of the training data. This technique helps the model generalize better to different voice patterns and improves its robustness.
  - **Conversion of Audio Files to CSV**: The audio files in the dataset were converted into a CSV file. Each row in the CSV file represents an audio file, with the 'filename' column storing the file names and the 'class' column storing the corresponding class names. This conversion simplifies data organization and allows for easier data handling and manipulation.
  - **Assignment of Folds for Cross-Validation**: The samples in the CSV file were assigned fold values for performing cross-validation. The 'fold' column was added to the CSV file, indicating which fold each sample belongs to based on the specified parameters such as the number of splits, shuffling, and random state. This assignment of folds ensures that each sample is included in both training and validation sets across different folds, enabling fair evaluation and mitigating overfitting.

The dataset can be accessed [here](https://drive.google.com/file/d/1koPDoUV0mSWXLva9gJtgCavo-agtyoqo).
## Model Selection

As mentioned in the introduction, the process of selecting the appropriate machine learning models for voice and handwriting classification tasks in the Calis application involved careful consideration and evaluation of various options. The chosen models were selected based on their suitability and performance in addressing the specific classification objectives. Let's explore the models in more detail:
### Handwriting Classification Model

For the handwriting classification task, we opted for a Convolutional Neural Network (CNN) model. CNNs have proven to be highly effective in image classification tasks, making them an ideal choice for accurately classifying handwritten characters. The CNN model takes advantage of its ability to capture spatial relationships and patterns within the input images, enabling it to distinguish between lowercase alphabets, uppercase alphabets, and digits. Through training the CNN model on a dataset of handwritten characters, we ensure that it learns the necessary features and achieves precise classification performance.
### Voice Classification Model

After exploring several options, we determined that adapting the pre-trained YAMNet model would be the most suitable approach for voice classification. YAMNet is a deep learning model specifically designed for sound classification tasks and has shown excellent performance in various audio classification domains. By fine-tuning YAMNet on our specific voice dataset and modifying its output classes, we can leverage its pre-learned representations to achieve accurate classification of voice recordings into the desired categories: alphabets, words, and sentences. This transfer learning approach allows us to benefit from the expertise of the pre-trained model and adapt it to our specific context, ensuring high-quality voice classification.

## Tutorial: Running the Notebook in Google Colab

This tutorial provides a step-by-step guide to running the notebook in Google Colab from the Calis-Top-10/Calis_ML repository. By following these instructions, you will be able to execute all the cells in the notebook and access the resulting TensorFlow Lite (TFLite) file.
### Prerequisites
To follow this tutorial, you need to have the following:
- A Google account
- Basic knowledge of using Google Colab and Jupyter notebooks

### Step 1: Opening the Notebook in Google Colab
1. Open your web browser and go to [Google Colab](https://colab.research.google.com).
2. Click on the "GitHub" tab in the dialog box that appears when you open Google Colab.

![Google Colab GitHub tab](https://i.imgur.com/V8fx8Oe.png)

3. In the "Enter a GitHub URL or search by organization or user" field, enter the URL of the Calis-Top-10/Calis_ML repository: `https://github.com/Calis-Top-10/Calis_ML`.
4. Press Enter or click on the search icon.
5. The repository will be displayed. Click on the notebook file (with the .ipynb extension) that you want to run.

### Step 2: Running the Notebook
1. Once the notebook is loaded, you will see the code cells and their contents.
2. To run a code cell, click on the cell to select it.
3. Press Shift + Enter or click on the Play button on the left side of the cell to execute it.
4. Continue executing the cells one by one until you have run all the cells in the notebook. Make sure to follow any instructions or guidelines provided within the notebook.

### Step 3: Accessing the TFLite File
1. After running all the cells, locate the TFLite file in the file explorer panel on the left side of the Colab interface.
2. Click on the folder icon in the left sidebar to expand the file explorer panel.

![Google Colab File Explorer](https://i.imgur.com/lul1zNq.png)

3. Navigate to the directory where the TFLite file is saved.
4. Once you find the TFLite file, you can click on it to download it to your local machine.

Congratulations! You have successfully run the notebook in Google Colab and accessed the TFLite file. You can now use the TFLite file for further deployment or integration into your application.

## Model Performance
### Handwriting Classification
#### Lowercase Alphabets
![Handwriting Classification: Lowercase Alphabets Accuracy](https://i.imgur.com/iicGKSh.png)
#### Uppercase Alphabets
![Handwriting Classification: Uppercase Alphabets Accuracy](https://i.imgur.com/InkjDHX.png)
#### Digits
![Handwriting Classification: Digits Accuracy](https://i.imgur.com/ggZbmHv.png)
### Voice Classification
![Voice Classification accuracy](https://i.imgur.com/HzOfdFm.png)
## References
- [Sound classification with YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet)
- [Pydub](https://github.com/jiaaro/pydub/blob/master/API.markdown)
- [Scikit-learn](https://scikit-learn.org/stable/modules/classes.html)
- [Numpy](https://numpy.org/doc/1.23/numpy-ref.pdf)
- [Pandas](https://pandas.pydata.org/docs/reference/index.html)
- [OS](https://docs.python.org/3/library/os.html)
- [Random](https://docs.python.org/3/library/random.html)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/)
- [TensorFlow I/O](https://www.tensorflow.org/io/api_docs/python/tfio)
- [PySoundFile](https://pysoundfile.readthedocs.io/en/latest/)
- [Matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_summary.html)
- [Shutil](https://docs.python.org/3/library/shutil.html)
- [Glob](https://docs.python.org/3/library/glob.html)
- [OpenCV](https://docs.opencv.org/4.7.0/)
