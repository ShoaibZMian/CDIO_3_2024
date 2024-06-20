import os
import cv2

dir_test = r'image_detection/big dataset/test/images'
dir_train = r"image_detection/big dataset/train/images"
dir_valid = r"image_detection/big dataset/valid/images"
print(f"Test: {len([entry for entry in os.listdir(dir_test) if os.path.isfile(os.path.join(dir_test, entry))])}")
print(f"Train: {len([entry for entry in os.listdir(dir_train) if os.path.isfile(os.path.join(dir_train, entry))])}")
print(f"Valid: {len([entry for entry in os.listdir(dir_valid) if os.path.isfile(os.path.join(dir_valid, entry))])}")

img = cv2.imread("image_detection/big dataset/train/images/WIN_20240313_09_24_04_Pro_mp4-0000_jpg.rf.95f5b52710fb925b77144781258b6dc6.jpg")
print(f"Picture shape: {img.shape}")