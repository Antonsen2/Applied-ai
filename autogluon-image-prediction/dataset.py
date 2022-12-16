import os
import shutil

def generate_csv(target_csv_file, src_image_dir, label_mappings):
    lines = []
    for root, _, files in os.walk(src_image_dir):
        for file in files:
            file_name = f"{root}/{file}".replace("\\", "/")
            label = get_label(file_name, label_mappings)
            lines.append(f"{file_name},{label}")

    with open(target_csv_file, "w") as f:
        f.write(f"image,label\n")
        f.write("\n".join(lines))

def generate_zip(target_file, src_dir):
    base_name, archive_format = target_file.split(".")
    shutil.make_archive(base_name, archive_format, src_dir)

def get_label(file_name, label_mappings):
    for search_str, label in label_mappings.items():
        if search_str in file_name:
            return label

    raise Exception(f"Cannot determine label for file: {file_name} using mappings: {label_mappings}")

if __name__ == "__main__":
    label_mappings = { "fire/fire": 0, "nofire/forest": 1 }
    generate_csv("dataset/test.csv", "dataset/test", label_mappings)
    generate_csv("dataset/train.csv", "dataset/train", label_mappings)

    generate_zip("target/autogluon_dataset.zip", "dataset")
