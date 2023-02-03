# Wildfire AI

## Introduction
The project aims to detect wildfires from users both in real-time and by uploading images to the website. The project uses one model for object detection and object localization and one model for image recognition. The models are trained on a dataset containing forest fires and non fire images. The goal of the project is to use drones and monitoring stations to identify areas affected by wildfires and alert relevant authorities.

## Technical Details
The image recognition model is built using a <b>Convolutional Neural Network(CNN)</b>. It stands out for its ways to process pixel data, through convolutional layers before output the result. It can also automatically detect features from input data, which minimise the human effort.
The model was selected for its ease of use and strong performance in initial testing..<br>
The object detection model is a <b>Faster R-CNN</b>, capable of localizing fire and smoke in an image and generating bounding boxes around these features. This allows the system to quickly identify areas affected by wildfire, improving the speed and accuracy of response efforts.

## Usage




## Contributing
Fork the project and pip install requrements.txt. To get full dataset please contact any contributor.<br>
To run docker containers do <code>docker-compose up --build</code> in root directory.
