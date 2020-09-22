# Data Scientist Nanodegree

## Dog Breed Identification App
### Dog Breed Classifier
   
## Table of Contents

- [Project Overview](#overview)
- [Project Components](#components)  
  - [Dog_app.ipynb](#notebook)
  - [Flask Web App](#flask)
- [Running](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Conclusion](#conclusion)
- [Files](#files)
- [Software Requirements](#sw)
- [Credits and Acknowledgements](#credits)

***

<a id='overview'></a>

## 1. Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the Data Scientist Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, our algorithm will identify the dog breed. If supplied an image of a human, the code will identify the resembling dog breed.

[Here](#eg) are a few screenshots of the web app.

<a id='main'></a>

## 2. Project Components

There are two components of this project:

<a id='notebook'></a>

### 2.1. Dog_app.ipynb Notebook

In this notebook we will do the following:
- Import Datasets
- Detect Humans
- Detect Dogs
- Create a CNN to Classify Dog Breeds (from Scratch)
- Use a CNN to Classify Dog Breeds (using Transfer Learning)
- Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Write our Algorithm
- Test our Algorithm

<a id='flask'></a>

### 2.2. Flask Web App

<a id='eg'></a>

Running [this command](#com) **from web directory** will start the web app where users can enter their query, i.e., a request message sent during a natural disaster.

**_Screenshot 1_**

![master](img/main.png)

What the app will do is that it will classify the text message into categories so that appropriate relief agency can be reached out for help.

<a id='run'></a>

## 3. Running

There are three steps to get up and runnning with the web app if you want to start from ETL process.

<a id='cleaning'></a>

### 3.1. Data Cleaning

**Go to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

_DisasterResponse.db_ already exists in _data_ folder but the above command will still run and replace the file with same information. 

<a id='training'></a>

### 3.2. Training Classifier

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will same information.

<a id='starting'></a>

### 3.3. Starting the web app

Now that we have cleaned the data and trained our model. Now it's time to see the prediction in a user friendly way.

**Go the app directory** and run the following command:

<a id='com'></a>

```bat
python app.py
```

This will start the web app and will direct you to a URL where you can enter messages and get classification results for it.

**_Screenshot 2_**

![master](img/charts.png)

<a id='conclusion'></a>

## 4. Conclusion

As we will notice, the model need more enhancements on its accuracy. This will require improving the algorithm to give better results at a larger scale or in a production environment. The following should be considered:

##### 1. More variety of breeds is required to enhance the model prediction for humans
##### 2. Improve the model to be able to classify pictures with noise
##### 3. Increasing number of epoch might help as well
##### 4. More images as per classes of dog, will help to improve model's accuracy

<a id='files'></a>

## 5. Files

<pre>
.
├── app
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP│   
│   └── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# DATA TO PROCESS
│   ├── disaster_messages.csv---------# DATA TO PROCESS
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── img-------------------------------# PLOTS FOR USE IN README AND THE WEB APP
├── models
│   └── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK

</pre>

<a id='sw'></a>

## 6. Software Requirements

This project uses **Python 3.7** and the necessary libraries are mentioned in _requirements.txt_.

```bat
pip3 install -r requirements.txt
```

<a id='credits'></a>

## 7. Credits and Acknowledgements <a name='licensing'></a>

Must give credit to [Udacity](https://www.udacity.com/courses/all) for creating this beautiful learning experience.  
Also, credit to [pythonanywhere] (https://www.pythonanywhere.com) for hosting my web app for free!
Find the Licensing for the data and other descriptive information from [Figure-eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/).
