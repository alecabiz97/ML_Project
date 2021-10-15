# ML_Project

## Project Files:

### icub_datasets.py

In this file we have two classes, iCubWorld28 and iCubWorld7 that inherit from torch.utils.data.Dataset and are implemented
to manage the homonyms datasets. We overrided the method __ init __, __ len __ and __ getitem __.

### main.py

Doesn't do anything (to be deleted).

### utils.py
In this file we implemented two functions that we use in our code, imshow and split_train_validation, respectively used
to show a tensor image and split randomly the data between train and validation.

### train_phase.py
In this file we implemented the training phase where we used some pretrained models adapting them to our dataset (finetuning).
We decided to use three different models as well as three different learning rates (Squeezenet 1.1, Resnet 152 and Densenet 161)
(0.01, 0.001, 0.0001). We trained the model using both 7 and 28 labels on 50 epochs and after that we saved our models and
the loss and accuracy results.

### test_phase.py
In this file we implemented the testing phase for our models, we loaded the models saved after the training phase and
evaluated the accuracy for each of them with respect to the test data. As for the training phase we saved our results for
both 7 and 28 classes and added the confusion matrix associated to see better the accuracy results.

### test_webcam.py
In this file we implemented a real-time test for our models, we load a selected model from the ones that we
saved before and then we use it to recognize some objects shown on the webcam. 

### plotting.py
In this file we implemented our plotting part to see the models together so that we can see the associated accuracy
for each of them and the behavior of the loss and accuracy and the test results as a histogram. Also in this
case we plot both the case with 7 classes and 28 classes.

### attack_testing_secml.py
In this file we executed an adversarial attack against a subset of our dataset based on the secml library, using 
foolbox to launch a prepared perturbation attack in order to reduce the accuracy of our models.
We launched the attack on three models that we choose with respect to the accuracy that the model obtained using a 
different learning rate each time. The chosen models are the Squeezenet 1.0, the Resnet 152 and the Densenet 161 with
learning rates of 0.01, 0.01 and 0.001 respectively. The results are saved on a unique plot in a pdf file.
