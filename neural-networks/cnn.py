import numpy as np
import keras

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

from utilities.nn import one_hot_encode


class CNN:
    """Initial Convolutional Neural Network from Keras"""
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

        self.weight_decay = 1e-4
        self.num_classes = 10

        self.model = self.model_set_up()
        self.model.summary()

    @staticmethod
    def lr_schedule(epoch: int) -> float:
        """Method to schedule the learning rate"""
        if epoch > 75:
            return 0.0005
        elif epoch > 100:
            return 0.0003
        return 0.001

    def model_set_up(self) -> keras.Sequential:
        """Set up the model layers"""
        model = Sequential()

        model.add(Conv2D(
            32,
            (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(self.weight_decay),
            input_shape=self.X.shape[1:]
        ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(
            32,
            (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(self.weight_decay)
        ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(
            64,
            (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(self.weight_decay)
        ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(
            64,
            (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(self.weight_decay)
        ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(
            128,
            (3,3), 
            padding='same',
            kernel_regularizer=regularizers.l2(self.weight_decay)
        ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(
            128,
            (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(self.weight_decay)
        ))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def train(self) -> None:
        """Main method. Training phase"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=104, test_size=0.4, shuffle=True
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, random_state=104, test_size=0.2, shuffle=True
        )

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        mean = np.mean(X_train,axis=(0, 1, 2, 3))
        std = np.std(X_train,axis=(0, 1, 2, 3))
        X_train = (X_train-mean) / (std+1e-7)
        X_test = (X_test-mean) / (std+1e-7)

        y_train = one_hot_encode(y_train)
        y_test = one_hot_encode(y_test)

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(X_train)

        batch_size = 64
        opt_rms = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
        self.model.fit_generator(
            datagen.flow(
                X_train,
                y_train,
                batch_size=batch_size
            ),
            steps_per_epoch=X_train.shape[0] // batch_size,
            epochs=125,
            verbose=1,
            validation_data=(X_val,y_val),
            callbacks=[LearningRateScheduler(self.lr_schedule)]
        )

    # TODO: Change save functionality from JSON to binary
    def save(self, model_name: str) -> None:
        """Save model to JSON and save the model weights"""
        model_json = self.model.to_json()
        with open(f'{model_name}.json', 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        self.model.save_weights('model.h5')
