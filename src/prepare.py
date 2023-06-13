import os
import pandas as pd


TABLE_PATH = f"../data/prepared"
DATASET_PATH = f"../data/raw"


def get_files_and_labels(source_path):
    images = []
    labels = []
    for label in os.listdir(source_path):
        paths = [f"{source_path}/{label}/{image}" for image in os.listdir(f"{source_path}/{label}")]
        images.extend(paths)
        labels.extend([label for _ in range(len(paths))])
    return images, labels


def save_as_csv(filenames, labels, destination):
    data_dictionary = {"path": filenames, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)
    data_frame.to_csv(destination, index=False)


def main():
    samples = list(filter(lambda file_or_dir: os.path.isdir(f"{DATASET_PATH}/{file_or_dir}"),
                          os.listdir(DATASET_PATH)))
    for sample in samples:
        files_paths, files_labels = get_files_and_labels(f"{DATASET_PATH}/{sample}")
        save_as_csv(files_paths, files_labels, f"{TABLE_PATH}/{sample}.csv")


if __name__ == "__main__":
    main()
