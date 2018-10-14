# Teachable Machine Dinosaur

Teachable Dino was created by two masters students for [MHacks 2018](https://twitter.com/mhacks?lang=en).

Inspired by the idleness in a grad student's life which is accompanied by staring into laptops for hours, we decided to explore ways of incorporating leisure phycial activity into their schedule. Teachable Dino hopes to save you from the office syndrome and encourage you to take a few short breaks to train him. 

We have combined [chrome's popular dino] chrome://dino/ with Google's [Teachable Machine Experiment] (https://teachablemachine.withgoogle.com/) . By using your webcam you can make gestures to train three actions : JUMP, DUCK and RUN. You can then play the game by using those gestures instead of the traditional keystrokes. 

How to Play:

1) Open How to [Train Your Dino](https://www.howtotrainyourdino.com/) in your web browser. 
2) Allow access to the webcam. (Note none of the images are stored, everything happens on your local browser)
3) Click on JUMP to record an image of the corresponding gesture. Number of clicks equals the number of times trained, for better results please train at least 50 times. 
4) Click on DUCK to record an image of the corresponding gesture. Number of clicks equals the number of times trained, for better results please train at least 50 times. 
5) Click on JUST RUN to record an image of the corresponding gesture. Number of clicks equals the number of times trained, for better results please train at least 50 times. 
6) Now,play! You can control the dino by your trained gestures. 

Things to do, future iterations:

1) Not start the game immediately until training is done. 
2) Add other versions of chrome dino.
3) Add music to accompanying the game. 

Watch a demo here.


Technical Explorations: 

We referred to [tensorflow.js](https://github.com/tensorflow/tfjs-models) which was used to create projects like [Teachable Machine](https://teachablemachine.withgoogle.com/). The code shows how you can create a KNN classifier that can be trained live in the browser on a webcam image. It is intentionally kept very simple so it can provide a starting point for new projects like these.

Behind the scenes, the image from the webcam is being processed by an activation of [MobileNet](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet). This network is trained to recognize all sorts of classes from the imagenet dataset, and is optimized to be really small, making it useable in the browser. Instead of reading the prediction values from the MobileNet network, we instead take the second to last layer in the neural network and feed it into a KNN ([k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)) classifier that allows you to train your own classes. 

The benefit of using the MobileNet model instead of feeding the pixel values directly into the KNN classifier is that we use the high level abstractions that the neural network has learned in order to recognize the Imagenet classes. This allows us with very few samples to train a classifier that can recognize things like smiles vs frown, or small movements in your body. This technique is called [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning).

Tensorflow has a built in model for doing this. It's called the [KNN Classifier](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier), and this boilerplate code shows how to easily use it.

## Use code
To use the code, first install the Javascript dependencies by running  

```
npm install
```

Then start the local budo web server by running 

```
npm start
```

This will start a web server on [`localhost:9966`](http://localhost:9966). Try and allow permission to your webcam, and add some examples by holding down the buttons. 

## Quick Reference - KNN Classifier
A quick overview of the most important function calls in the tensorflow.js [KNN Classifier](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier).

- `knnClassifier.create()`: Returns a KNNImageClassifier.

- `.addExample(example, classIndex)`: Adds an example to the specific class training set.

- `.clearClass(classIndex)`: Clears a specific class for training data.

- `.predictClass(image)`: Runs the prediction on the image, and returns an object with a top class index and confidence score. 

See the full implementation [here](https://github.com/tensorflow/tfjs-models/blob/master/knn-classifier/src/index.ts)

## Quick Reference - MobileNet
A quick overview of the most important function calls we use from the tensorflow.js model [MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet).

- `.load()`: Loads and returns a model object.

- `.infer(image, endpoint)`: Get an intermediate activation or logit as Tensorflow.js tensors. Takes an image and the optional endpoint to predict through.

See the full implementation [here](https://github.com/tensorflow/tfjs-models/blob/master/mobilenet/src/index.ts)
