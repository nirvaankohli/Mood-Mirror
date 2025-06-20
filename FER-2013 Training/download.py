import kagglehub
import sys

path = kagglehub.dataset_download("msambare/fer2013", sys.path[0], force_download=True)

print("Path to dataset files(Even though the dataset should have been downloaded to this directory):", path)