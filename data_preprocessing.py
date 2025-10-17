import pandas as pd
import numpy as np
import cv2
import keras.models as M
import keras.layers as L
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
import keras
from skimage.transform import resize
import tifffile
import matplotlib.pyplot as plt
import tifffile as tiff
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize


folder_path = "/kaggle/input/ant-hv-may-oct"
file_list = os.listdir(folder_path)
num_images = len(file_list)
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.ceil(num_images / num_rows))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
for i, file_name in enumerate(file_list):
    if file_name.lower().endswith('.tiff') or file_name.lower().endswith('.tiff'):
        file_path = os.path.join(folder_path, file_name)
        image = tiff.imread(file_path)
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes[row_idx, col_idx].imshow(image, cmap='gray')
        axes[row_idx, col_idx].axis('off')
plt.tight_layout()
plt.show()

def normalize_backscattering(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = 255 * (data - min_value) / (max_value - min_value)
    normalized_data = normalized_data.astype(np.uint8)
    return normalized_data

folder_path = '/kaggle/input/ant-hv-may-oct' 

norm_folder = '/kaggle/working/norm_data' 

if not os.path.exists(folder_path):
    print(f"Folder not found: {folder_path}")
else:
    if not os.path.exists(norm_folder):
        os.makedirs(norm_folder)
  for filename in os.listdir(folder_path):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            backscattering_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            normalized_data = normalize_backscattering(backscattering_data)
            min_value = np.min(normalized_data)
            max_value = np.max(normalized_data)
            print(f"File: {filename}")
            print("Minimum value:", min_value)
            print("Maximum value:", max_value)
            print("\n")
            new_filename = "normalized_" + filename
            new_file_path = os.path.join(norm_folder, new_filename)
            cv2.imwrite(new_file_path, normalized_data)
print("Normalization and saving complete!")

def convert_to_db(image):
    image[image == 0] = 1e-9
    min_clip_value = 1e-9
    max_clip_value = 1e9
    image = np.clip(image, min_clip_value, max_clip_value)
    with np.errstate(divide='ignore', invalid='ignore'):
        db_image = 10 * np.log10(image)
    db_image[np.isinf(db_image)] = np.nan
    return db_image

def generate_mask_lib(db_image):
    threshold_ice_free = [-np.inf,0] 
    threshold_firstyearice = [0, 4] 
    threshold_ice_bergs = [4,6] 
    threshold_multiyearice = [6, np.inf] 
    

    mask_ice_free = (db_image >= threshold_ice_free[0]) & (db_image <= threshold_ice_free[1])
    mask_firstyearice = (db_image >= threshold_firstyearice[0]) & (db_image <= threshold_firstyearice[1])
    mask_ice_bergs = (db_image >= threshold_ice_bergs[0]) & (db_image <= threshold_ice_bergs[1])
    mask_multiyearice = (db_image >= threshold_multiyearice[0]) & (db_image <= threshold_multiyearice[1])
    

    mask_lib = {
        'Ice Free': mask_ice_free.astype(int),
        'First-Year Ice': mask_firstyearice.astype(int),
        'Ice bergs': mask_ice_bergs.astype(int),
        'Multi-Year Ice': mask_multiyearice.astype(int),
    }

    return mask_lib

def map_backscatter_to_label(backscatter):
    if -np.inf <= backscatter <= 0:
        return 0  
    elif 0 <= backscatter <= 4:
        return 1  
    elif 4 <= backscatter <= 7:
        return 2 
    elif 7<= backscatter <= np.inf:
        return 3 
    else:
        return -1 

folder_path = "/kaggle/working/norm_data"
backscatter_values = []
y = []
X = []
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    image_array = tifffile.imread(image_path)

    if np.any(np.isnan(image_array)) or np.any(np.isinf(image_array)):
        print(f"Invalid values found in {filename}. Skipping...")
        continue

    db_image = convert_to_db(image_array)
    print(f"Filename: {filename}, Decibel value: {db_image}")
    
    backscatter_value = np.nanmean(db_image)

    print(f"Filename: {filename}, Backscatter Value: {backscatter_value}")

    label_encoded = map_backscatter_to_label(backscatter_value)

    print(f"Encoded Label: {label_encoded}")

    if label_encoded != -1:
        y.append(label_encoded)

        resized_image = resize(image_array, (150, 150))

        X.append(resized_image)

X = np.array(X)
y = np.array(y)

print("Encoded Labels:")
print(y)

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

save_dir = '/kaggle/working/augmented_img'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print('Performing data augmentation')
augmented_images = []
augmented_labels = []
num_augmented_samples = 3

for i in range(len(X)):
    img = X[i]
    label = y[i]

    img = np.expand_dims(img, axis=-1)
    img_augmented_gen = datagen.flow(np.expand_dims(img, axis=0), batch_size=1, shuffle=False)

    for j in range(num_augmented_samples):
        img_augmented = img_augmented_gen.next()[0]
        img_augmented = resize(img_augmented, (150, 150))

        augmented_images.append(img_augmented)
        augmented_labels.append(label)
        img_save_path = os.path.join(save_dir, f'image_{i}_{j}.tiff')
        plt.imsave(img_save_path, img_augmented.squeeze(), cmap='gray')

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

augmented_labels = np.expand_dims(augmented_labels, axis=-1)
print('Augmented images shape:', augmented_images.shape)
print('Augmented labels shape:', augmented_labels.shape)
print('Augmented images saved in:', save_dir)

