import os
import zipfile
import numpy as np


ARCHIVES_PATH = "../../../res/archive/"
ANGRY = ['angry', 'Angry']
DISGUST = ['disgusted', 'disgust', 'Disgust']
FEAR = ['fearful', 'fear', 'Fear']
HAPPY = ['happy', 'Happy']
NEUTRAL = ['neutral', 'Neutral']
SAD = ['sad', 'Sad']
SURPRISED = ['surprised', 'surprise', 'Surprise']
EMOTION_CLASSES = [ANGRY, DISGUST, FEAR, HAPPY, NEUTRAL, SAD, SURPRISED]


def get_archives_labels(archives_path):
    # init list of archives labels
    archives_labels = []
    # explore each archive
    for archive_name in os.listdir(archives_path):
        # read archive
        with zipfile.ZipFile(f"{archives_path}/{archive_name}", 'r') as z:
            # get unique labels of specific archive
            label_names = list(set([os.path.dirname(frame).split('/')[-1] for frame in z.namelist()]))
            label_names.sort()
            # add archive and its targets to list
            archives_labels.append({archive_name: label_names})

    return archives_labels


def display_archives_labels(archives_path):
    # get archives and their targets
    archives_labels = get_archives_labels(archives_path)
    # display each one
    for archive_labels in archives_labels:
        print(
            f"Archive {list(archive_labels.keys())[0]}:\n"
            f"{list(archive_labels.values())[0]}\n"
            f"Length: {len(list(archive_labels.values())[0])}\n"
        )


def get_frame_counts_tree(archives_path):
    archive_names = os.listdir(archives_path)
    for archive_name in archive_names:
        with zipfile.ZipFile(f"{archives_path}/{archive_name}", 'r') as z:
            label_names = list(set([os.path.dirname(frame) for frame in z.namelist()]))
            label_names.sort()
            print(f"Archive {archive_name}:")
            label_counts = []
            for label_name in label_names:
                label_counts.append(len([frame for frame in z.namelist() if label_name in frame]))
                print(f"Length of {label_name}:\t{label_counts[-1]}")
            test_label_counts = np.array(label_counts[:7])
            train_label_counts = np.array(label_counts[7:])
            if archive_name != 'rating-opencv-emotion-images.zip':
                print(test_label_counts/train_label_counts)
            else:
                print(train_label_counts/test_label_counts)
            print()


if __name__ == "__main__":
    # common labels: angry, disgust, fear, happy, neutral, sad, surprise
    # display_archives_labels(ARCHIVES_PATH)

    # emotion-detection-fer.zip - ~0.25
    # fer2013.zip - ~0.25
    # rating-opencv-emotion-images.zip - ~0.125
    get_frame_counts_tree(ARCHIVES_PATH)
