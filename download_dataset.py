import os
import requests
import zipfile

# Define the dataset URL (official DeepFashion2 link)
# DATASET_URL = "http://vision.cs.stonybrook.edu/~liu/projects/DeepFashion2/Dataset.zip"
SAVE_PATH = "deepfashion2.zip"
EXTRACT_PATH = "deepfashion2_dataset"

# Function to download the dataset
def download_dataset(url, save_path):
    print(f"Downloading DeepFashion2 dataset from {url}...")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("Download completed!")

# Function to extract the dataset
def extract_dataset(zip_path, extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Dataset extracted to {extract_path}")

# Run the functions
if not os.path.exists(SAVE_PATH):
    download_dataset(DATASET_URL, SAVE_PATH)
else:
    print("Dataset already downloaded.")

if not os.path.exists(EXTRACT_PATH):
    extract_dataset(SAVE_PATH, EXTRACT_PATH)
else:
    print("Dataset already extracted.")
