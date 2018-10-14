// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;


class Main {
    constructor() {
        // Initiate variables
        this.infoTexts = [];
        this.training = -1; // -1 when no class is being trained
        this.videoPlaying = false;

        // Initiate deeplearn.js math and knn classifier objects
        this.bindPage();

        const code = {
            0: {"btext": "Jump", "code": 38},
            1: {"btext": "Duck", "code": 40},
            2: {"btext": "Just Run", "code": ""}
        };

        // Create video element that will contain the webcam image
        const div = document.createElement('div');
        div.style.textAlign = 'center';
        //div.classList.add('section__container');
        this.video = document.createElement('video');
        this.video.setAttribute('autoplay', '');
        this.video.setAttribute('playsinline', '');
        this.video.style.alignSelf = 'center';

        // Add video element to DOM
        div.appendChild(this.video);
        document.body.appendChild(div);

        // Create training buttons and info texts
        for (let i = 0; i < NUM_CLASSES; i++) {
            const div = document.createElement('div');
            div.style.alignSelf = 'center';
            div.style.textAlign = 'center';
            document.body.appendChild(div);
            div.style.marginBottom = '10px';

            // Create info text
            const infoText = document.createElement('span')
            infoText.innerText = "You have trained me 0 times to  ";
            div.appendChild(infoText);
            this.infoTexts.push(infoText);

            // Create training button
            const button = document.createElement('button')
            button.innerText = code[i].btext;
            button.classList.add("mdl-button");
            button.classList.add("mdl-js-button");
            button.classList.add("mdl-button--raised");
            button.classList.add("mdl-button--colored");
            button.classList.add('mdl-button--primary');
            div.appendChild(button);

            // Listen for mouse events when clicking the button
            button.addEventListener('mousedown', () => this.training = i);
            button.addEventListener('mouseup', () => this.training = -1);


        }


        // Setup webcam
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then((stream) => {
                this.video.srcObject = stream;
                this.video.width = IMAGE_SIZE;
                this.video.height = IMAGE_SIZE;

                this.video.addEventListener('playing', () => this.videoPlaying = true);
                this.video.addEventListener('paused', () => this.videoPlaying = false);
            })
    }

    async bindPage() {
        this.knn = knnClassifier.create();
        this.mobilenet = await mobilenetModule.load();

        this.start();
    }

    start() {
        if (this.timer) {
            this.stop();
        }
        this.video.play();
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }

    stop() {
        this.video.pause();
        cancelAnimationFrame(this.timer);
    }

    async animate() {
        if (this.videoPlaying) {
            // Get image data from video element
            const image = tf.fromPixels(this.video);

            let logits;
            // 'conv_preds' is the logits activation of MobileNet.
            const infer = () => this.mobilenet.infer(image, 'conv_preds');

            // Train class if one of the buttons is held down
            if (this.training != -1) {
                logits = infer();

                // Add current image to classifier
                this.knn.addExample(logits, this.training)
            }

            const code = {
                0: {"key": "up", "code": 38},
                1: {"key": "down", "code": 40},
                2: {"key": "nothing", "code": ""}
            };

            const numClasses = this.knn.getNumClasses();
            if (numClasses > 0) {

                // If classes have been added run predict
                logits = infer();
                const res = await this.knn.predictClass(logits, TOPK);

                let keyUpFunction = function(code) {
                    let e2 = document.createEvent('Event');
                    e2.initEvent('keyup', true, true);
                    e2.keyCode = code[parseInt(res.classIndex)]['code'];
                    document.dispatchEvent(e2);
                }

                if ((res.classIndex !== 2) || (typeof(res.classIndex) !== undefined)) {

                    let e = document.createEvent('Event');
                    e.initEvent('keydown', true, true);
                    e.keyCode = code[parseInt(res.classIndex)]['code'];
                    document.dispatchEvent(e);
                    setTimeout(keyUpFunction, 1000, code);
                    //keyUpFunction(code);
                }

                for (let i = 0; i < NUM_CLASSES; i++) {

                    // The number of examples for each class
                    const exampleCount = this.knn.getClassExampleCount();

                    // Make the predicted class bold
                    if (res.classIndex == i) {
                        this.infoTexts[i].style.fontWeight = 'bold';
                    } else {
                        this.infoTexts[i].style.fontWeight = 'normal';
                    }

                    // Update info text
                    if (exampleCount[i] > 0) {
                        this.infoTexts[i].innerText = `You have trained me ${exampleCount[i]} times - ${res.confidences[i] * 100}% `
                    }
                }
            }

            // Dispose image when done
            image.dispose();
            if (logits != null) {
                logits.dispose();
            }
        }
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }
}

window.addEventListener('load', () => new Main());