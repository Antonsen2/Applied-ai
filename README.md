# Wildfire AI

## Introduction
The project aims to detect wildfires from users both in real-time and by uploading images to the website. The project uses one model for object detection and object localization and one model for image recognition. The models are trained on a dataset containing forest fires and non fire images. The goal of the project is to use drones and monitoring stations to identify areas affected by wildfires and alert relevant authorities.

## Table of contents
  * [Introduction](#introduction)
  * [Technical Details](#technical-details)
  * [Usage](#usage)
  * [Contribution](#contribution)

## Technical Details

The image recognition model is built using a <b>Convolutional Neural Network (CNN)</b>, a deep learning model designed for processing pixel data in images. The CNN uses convolutional layers to extract and process information from the input data, reducing the need for manual feature engineering by automatically detecting features from the input. The CNN was selected for this project due to its ease of use and strong performance in initial testing.

The object detection model is a <b>Faster R-CNN</b>, which stands for Region-based Convolutional Neural Network. The model <b>Faster R-CNN</b> is capable of localizing fire and smoke in an image and generating bounding boxes around these features. This allows the system to quickly identify areas affected by wildfire, improving the speed and accuracy of response efforts. In the context of this project, the Faster R-CNN is used to identify fires and smoke in an image and to draw bounding boxes around these features, allowing the system to quickly identify areas affected by wildfire and to improve the speed and accuracy of response efforts.


## Usage - Get started
1. Fork the project
2. Pip install requrements.txt 
3. To get full dataset please contact any contributor or try your own dataset.
4. The classification model is in the repository, but for object detection, you will to up load by yourself as it is too big for github. You can also contact any contributor for object detection model or train yourself in the research branch.
5. Add .env file which should contain 
    ```
    TOKEN=yourownpreferences
    LOG_LEVEL=INFO
    ```
6. Run docker containers in root directory.
   ```
   docker compose up --build
    ```
8. Go to  http://localhost:8000/classify to test the model

### Note

If you would like to re train/ checkout how we train the models you can find them in <b>research branch</b>


## Contribution
<a href="https://github.com/Antonsen2">Markus</a><br>
<a href="https://github.com/AndreasEliasson91">Andreas</a><br>
<a href="https://github.com/meDracula">Albin</a><br>
<a href="https://github.com/Alicia-Toom">Alicia</a><br>
<a href="https://github.com/mattiasbarth">Mattias</a><br>
