import argparse

import kagglehub
from kagglehub import KaggleDatasetAdapter


file_path = ""


def download(name, path):
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        name,
        path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset script.')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--data_dir', type=str, default='../StackMix-OCR-DATA')
    args = parser.parse_args()

    download(args.dataset_name, args.data_dir)
