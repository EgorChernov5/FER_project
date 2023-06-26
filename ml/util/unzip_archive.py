import os
import random
import zipfile
import numpy as np
import pandas as pd
import shutil


ARCHIVE_PATH = "../data/archive"
DATASET_PATH = "../data/raw"
FOLDERS_TO_LABELS = {
    'angry': 'angry',
    'anger': 'angry',
    'Angry': 'angry',
    'disgust': 'disgust',
    'disgusted': 'disgust',
    'Disgust': 'disgust',
    'fear': 'fear',
    'fearful': 'fear',
    'Fear': 'fear',
    'happy': 'happy',
    'happiness': 'happy',
    'Happy': 'happy',
    'neutral': 'neutral',
    'neutrality': 'neutral',
    'Neutral': 'neutral',
    'sad': 'sad',
    'sadness': 'sad',
    'Sad': 'sad',
    'surprise': 'surprise',
    'surprised': 'surprise',
    'Surprise': 'surprise'
}


def display_raw_samples(checked_sample='test'):
    samples = os.listdir(DATASET_PATH)
    labels = os.listdir(f"{DATASET_PATH}/train")
    labels_counts = dict.fromkeys(labels, 0)
    sample_labels_counts = dict.fromkeys(labels, 0)
    for sample in samples:
        for label in labels:
            labels_counts[label] += len(os.listdir(f"{DATASET_PATH}/{sample}/{label}"))
            if sample == checked_sample:
                sample_labels_counts[label] += len(os.listdir(f"{DATASET_PATH}/{checked_sample}/{label}"))

    for label in labels:
        print(f"Ratio of {checked_sample} {label}: {round(sample_labels_counts[label]/labels_counts[label], 3)}")
    print(labels_counts)


def display_archive_samples(archive_path, targets_count):
    with zipfile.ZipFile(archive_path, 'r') as z:
        label_names = list(set([os.path.dirname(frame) for frame in z.namelist()]))
        label_names.sort()
        print(f"Archive {archive_name}:")
        # counter for samples
        label_counts = []
        for label_name in label_names:
            label_counts.append(len([frame for frame in z.namelist() if label_name in frame]))
            print(f"Length of {label_name}:\t{label_counts[-1]}")
        # counter for test/train samples
        test_label_counts = np.array(label_counts[:targets_count])
        train_label_counts = np.array(label_counts[targets_count:])

        labels_counts = test_label_counts + train_label_counts
        test_label_ratio = [round(i, 3) for i in test_label_counts / labels_counts]
        print(test_label_ratio)
        print(labels_counts)


def check_extension(file_name):
    valid = ['png', 'jpg', 'jpeg']
    return file_name.split('.')[-1] in valid


def get_temp_dep(images_paths):
    # get all dirs
    dirs_paths = list(set([os.path.dirname(image_path) for image_path in images_paths]))
    # get temporary dependencies
    temp_dep = pd.DataFrame(columns=['path', 'label'])
    for dir_path in dirs_paths:
        # get folder of the image path
        folder = dir_path.split("/")[-1]
        # get label by folder
        try:
            label = FOLDERS_TO_LABELS[folder]
        except:
            print(f"There is no '{folder}' label")
            continue
        # get members by dir path
        members = pd.DataFrame(filter(lambda image_path: dir_path in image_path, images_paths), columns=['path'])
        # add paths to temporary df
        temp_dep = pd.concat([temp_dep, members], ignore_index=True)
        temp_dep = temp_dep.replace(np.nan, label)
    return temp_dep


def unzip_archive(source, destination):
    with zipfile.ZipFile(source, 'r') as z:
        # get all paths
        images_paths = list(filter(check_extension, z.namelist()))
        # get temporary dependencies
        temp_dep = get_temp_dep(images_paths)
        # get the total number of samples
        label_counts = temp_dep['label'].value_counts()
        # explore each label
        for label in label_counts.keys():
            # get number of images in the sample
            count = label_counts[label]
            # make proportions
            samples_sizes = {
                'train': int(count*0.6),
                'val': int(count*0.2),
                'test': count - int(count*0.6) - int(count*0.2)
            }
            # get sample paths
            members = temp_dep[temp_dep['label'] == label]['path']
            start = 0
            # explore each sample of label
            for sample in samples_sizes.keys():
                # get sample size
                sample_size = samples_sizes[sample]
                end = start + sample_size
                # get sample with definite size
                sample_members = members[start:end]
                # extract files
                z.extractall(path=f"{destination}/{sample}/{label}", members=sample_members)
                # replace each image path to required folder
                for member_path in sample_members:
                    file_name = member_path.split('/')[-1]
                    # dir_name = os.path.dirname(member_path)
                    os.replace(src=f"{destination}/{sample}/{label}/{member_path}",
                               dst=f"{destination}/{sample}/{label}/{random.randint(1, 100)}_{file_name}")
                # get empty folders which were extract with images
                empty_dirs = list(filter(lambda file_or_dir: os.path.isdir(f"{destination}/{sample}/{label}/{file_or_dir}"),
                                         os.listdir(f"{destination}/{sample}/{label}")))
                # delete unnecessary folders
                for empty_dir in empty_dirs:
                    shutil.rmtree(f"{destination}/{sample}/{label}/{empty_dir}")

                start += sample_size


if __name__ == "__main__":
    # display_raw_samples()
    archives_names = os.listdir(ARCHIVE_PATH)
    for archive_name in archives_names:
        # display_archive_samples(f"{ARCHIVE_PATH}/{archive_name}", targets_count=7)
        unzip_archive(f"{ARCHIVE_PATH}/{archive_name}", DATASET_PATH)
