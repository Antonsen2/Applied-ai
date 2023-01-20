import os
import uuid

import pandas as pd
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import path_expander

def download_train_test_data(zip_file_url, unzip_dir):
    #load_zip.unzip(zip_file_url, unzip_dir=unzip_dir)

    train_data = pd.read_csv(f"{unzip_dir}/train.csv")
    test_data = pd.read_csv(f"{unzip_dir}/test.csv")

    train_data["image"] = train_data["image"].apply(lambda x: path_expander(x, base_folder=unzip_dir))
    test_data["image"] = test_data["image"].apply(lambda x: path_expander(x, base_folder=unzip_dir))
    return train_data, test_data

if __name__ == "__main__":
    #zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/shopee.zip"
    #tmp_dir = os.path.abspath("./tmp/data")
    #tmp_dir = os.path.abspath("./dataset")
    model_dir = f"./tmp/model/{uuid.uuid4().hex}"

    #train_data, test_data = download_train_test_data(zip_file, tmp_dir)

    base_folder = os.path.abspath("./")
    train_data = pd.read_csv("./dataset/train.csv")
    test_data = pd.read_csv("./dataset/test.csv")

    train_data["image"] = train_data["image"].apply(lambda x: path_expander(x, base_folder=base_folder))
    test_data["image"] = test_data["image"].apply(lambda x: path_expander(x, base_folder=base_folder))
    
    predictor = MultiModalPredictor(label="label", path=model_dir)
    predictor.fit(
        train_data=train_data,
        time_limit=120, # seconds
    )

    scores = predictor.evaluate(test_data, metrics=["accuracy"])
    print(f"Top-1 test acc: {scores['accuracy']}")

    fire_image_path = f"{base_folder}/dataset/test/fire/fire-80.jpg"
    nofire_image_path = f"{base_folder}/dataset/test/nofire/forest-80.jpg"
    predictions = predictor.predict({'image': [fire_image_path, nofire_image_path]})
    print(predictions)
