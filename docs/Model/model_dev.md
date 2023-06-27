## Stages of development of a machine learning model for recognizing human emotions to improve the gaming industry:

### Постановка цели

Collecting statistical data on a person's emotional state can be useful for assessing the emotional appeal of the game and identifying strengths and weaknesses.
Developers will be able to conduct more targeted testing and make improvements to create games that evoke strong emotions and satisfaction among players.

### Data collection

**The training data was taken from the Kaggle website:**

* [Emotion Detection](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
* [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)  
* [Rating OpenCV Emotion Images](https://www.kaggle.com/datasets/juniorbueno/rating-opencv-emotion-images)  
* [Natural Human Face Images for Emotion Recognition](https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition)  
* [Micro_Expressions](https://www.kaggle.com/datasets/kmirfan/micro-expressions)  
* [Corrective re-annotation of FER - CK+ - KDEF](https://www.kaggle.com/datasets/sudarshanvaidya/corrective-reannotation-of-fer-ck-kdef)  

### Data preprocessing

Repositories have a different structure, so the functionality was implemented (utils.unzip_archive.py ) to unzip data into a single folder structure. Next, we bring the photos to a single format:

* 1 color channel;
* shape 48x48;
* the formation of approximately the same number of photos of samples.

### Model Selection

ДA simple convolutional neural network with 5 convolution layers and ReLU activation functions was implemented in baseline. For implementation, we use the TensorFlow framework.

**Structure:**
```python
tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu',
                               input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation='softmax'),
    ])
```

### Model Training

We use the prepared data for training. We divide the data into training, validation and test samples in a ratio of 60x20x20.

**Model parameters:**
```python
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

EPOCHS = 30
batch_size=100
```

**When learning, we get the following values:**

* Train acc: 0.916867196559906
* Val acc: 0.7075471878051758

### Evaluation and improvement of the model

The next stage is the visualization of graphs (loss, accuracy, confusion matrix), evaluation of the model based on test data.

Based on the test data, we received accuracy 0.7075471878051758, this does not suit us.

Based on the estimated data, we see that we have a fairly large bias on the train sample - this suits us, but when we look at the accuracy of our validation sample, we notice that we have a large variance of ~ 20%.

There are several ways we can fix this:

* Apply regularization;
* Increase the amount of data;
* Change the model structure.

### Implementation of the model

After successful evaluation and improvement of the model, we proceed to its implementation. We create an API so that it can interact with other systems using TenzorFlow Serving. We put our model in the image and run the image.

The model can be accessed at the following address - [http://ts-ml-service:8501/v1/models/fer_model:predict](http://ts-ml-service:8501/v1/models/fer_model:predict).

The transmitted data should look like this:
```json
{
  "instances": np.array(shape=[1, 48, 48, 1])
}
```