import os
import argparse


def clean_directory(directory, allowed_extensions):
    allowed_extensions = [ext.lower() for ext in allowed_extensions]
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() not in allowed_extensions:
                os.remove(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--extensions", nargs="+", required=True)
    args = parser.parse_args()
    allowed_extensions = [
        f"{ext if ext.startswith('.') else '.' + ext}".lower()
        for ext in args.extensions
    ]
    clean_directory(args.data_dir, allowed_extensions)
