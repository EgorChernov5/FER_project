import os
from pathlib import Path
import pandas as pd


REPO_DIR = Path(__file__).parent.parent


def display_sample_counts(csv_file):
    data = pd.read_csv(csv_file)
    print(data['label'].value_counts())


def get_files_and_labels(sample_path):
    images = []
    labels = []
    for labels_paths in sample_path.iterdir():
        for label_path in labels_paths.iterdir():
            images.append(str(label_path).split(str(REPO_DIR))[-1].replace('\\', '/')[1:])
            labels.append(str(label_path).split(os.path.sep)[-2])
    return images, labels


def save_as_csv(filenames, labels, destination):
    data_dictionary = {"path": filenames, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)
    data_frame.to_csv(destination, index=False)


def raw_to_prepared(raw_path: Path):
    for filesys_obj in raw_path.iterdir():
        if filesys_obj.is_dir():
            files_paths, files_labels = get_files_and_labels(filesys_obj)
            sample = str(filesys_obj).split(os.path.sep)[-1]
            save_as_csv(files_paths, files_labels, REPO_DIR / f"data/prepared/{sample}.csv")


def main():
    # raw_to_prepared(REPO_DIR / 'data/raw')
    display_sample_counts(REPO_DIR / 'data/prepared/test.csv')


if __name__ == "__main__":
    main()
