"""CNN model v2"""


import os.path
import pandas as pd
import tensorflow as tf

from keras.layers import Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from models.CNN.config import IMG_SIZE, COLOR_MODE, CLASS_MODE, CHECKPOINT_PATH, IMG_DIR, BATCH_SIZE, RANDOM_SEED

from sklearn.model_selection import train_test_split
from utilities.data import split_df, create_callback

RESIZE_AND_RESCALE = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(*IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])

TRAIN_GENERATOR = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)
TEST_GENERATOR = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

class ImageGenerator:
    def __init__(self, split: bool = False, cols: list = None, **kwargs) -> None:
        dataframe = self.set_up_df()
        if split:
            dataframe = split_df(dataframe, cols, **kwargs)

        train_df, test_df = train_test_split(
            dataframe, test_size=0.3, shuffle=True, random_state=42
        )

        self.train_images = TRAIN_GENERATOR.flow_from_dataframe(
            dataframe=train_df,
            x_col=cols[0] if cols else 'Filepath',
            y_col=cols[1] if cols else 'Label',
            target_size=IMG_SIZE,
            color_mode=COLOR_MODE,
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=RANDOM_SEED,
            subset='training'
        )
        self.validation_images = TRAIN_GENERATOR.flow_from_dataframe(
            dataframe=train_df,
            x_col=cols[0] if cols else 'Filepath',
            y_col=cols[1] if cols else 'Label',
            target_size=IMG_SIZE,
            color_mode=COLOR_MODE,
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=RANDOM_SEED,
            subset='validation'
        )
        self.test_images = TEST_GENERATOR.flow_from_dataframe(
            dataframe=test_df,
            x_col=cols[0] if cols else 'Filepath',
            y_col=cols[1] if cols else 'Label',
            target_size=IMG_SIZE,
            color_mode=COLOR_MODE,
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

    @staticmethod
    def set_up_df() -> pd.DataFrame:
        """Get all image path from IMG_DIR and label them from their directory name
        return:
            Pandas Dataframe, dataframe with filepaths and labels mapped
        """
        filepaths = list(IMG_DIR.glob(r'**/*.JPG')) + list(IMG_DIR.glob(r'**/*.jpg')) + list(IMG_DIR.glob(r'**/*.png'))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        df = pd.concat([filepaths, labels], axis=1)
        return df

class CNN:
    """CNN model used for Image Classification"""
    def __init__(self, split: bool = False, cols: list = None, **kwargs) -> None:
        self.history = None
        
        self.images = ImageGenerator(split, cols, **kwargs)
        self.X, self.inputs = self.setup()
        self.outputs = Dense(2, activation='softmax')(self.X)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    @staticmethod
    def setup() -> tuple:
        """Process X for model training based on pre-trained MobileNetV2 model
        return:
            tuple, X and inputs from pre-trained model
        """
        pretrained_model = MobileNetV2(
            input_shape=(250, 250, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        pretrained_model.trainable = False

        X = RESIZE_AND_RESCALE(pretrained_model.input)
        X = Dense(256, activation='relu')(pretrained_model.output)
        X = Dropout(0.2)(X)
        X = Dense(256, activation='relu')(X)
        X = Dropout(0.2)(X)

        return X, pretrained_model.input

    def train(self, _dir: str, log_dir: str) -> None:
        """Model train method"""
        checkpoint_callback = ModelCheckpoint(
            CHECKPOINT_PATH,
            save_weights_only=True,
            monitor='val_accuracy',
            save_best_only=True
        )
        early_stopping = EarlyStopping(
            monitor = "val_loss",
            patience = 5,
            restore_best_weights = True
        )

        self.model.compile(
            optimizer=Adam(0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            self.images.train_images,
            steps_per_epoch=len(self.images.train_images),
            validation_data=self.images.validation_images,
            validation_steps=len(self.images.validation_images),
            epochs=100,
            callbacks=[
                early_stopping,
                create_callback(_dir, log_dir),
                checkpoint_callback,
            ]
        )
