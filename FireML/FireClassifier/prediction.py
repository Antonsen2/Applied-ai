from tensorflow import keras

MODEL_PATH="model/model.h5"
MODEL = keras.models.load_model(MODEL_PATH)


def preprocess_image(data):
    return "something"


def model_predict(image) -> str:
    #pred = MODEL(image)
    return "fire"
