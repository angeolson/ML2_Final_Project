# File to download and save the project data

import os
import gdown
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()

url = 'https://drive.google.com/u/3/uc?id=16U5WG2Jo8mU-J2jpo3PCnsvAn2CF2hF0&export=download'
PATH = args.path
# PATH ='/home/ubuntu/Final-Project-Group4'
DATA_PATH = PATH + os.path.sep + 'Data'
os.chdir(DATA_PATH)
gdown.download(url, 'archive(4).zip', quiet=False)
os.system("unzip 'archive(4).zip'")
print('Done!')