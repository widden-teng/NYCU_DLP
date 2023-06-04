from PIL import Image
import numpy as np
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from dataset.evaluator import evaluation_model

# Load the big image
# file_name = 'output/test/output_epoch58.png'
file_name = 'output/new_test/output_epoch45.png'
big_image = Image.open(file_name)

# Get the size of the big image
width, height = big_image.size

# Define the size of each small image (excluding the black line)
small_image_size = (width // 8 - 2, height // 4 - 2)
# Create an empty list to store the cropped small images
small_images = []
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Loop over the 8 rows and 4 columns
for row in range(4):
    for col in range(8):
        # Calculate the coordinates for cropping
        left = col * (small_image_size[0] + 2) + 2
        top = row * (small_image_size[1] + 2) + 2
        right = left + small_image_size[0]
        bottom = top + small_image_size[1]

        # Crop the small image from the big image
        small_image = big_image.crop((left, top, right, bottom))

        # Convert the small image to numpy array
        small_image_tensor = transform(small_image)
        # Append the small image to the list
        small_images.append(small_image_tensor)

print(small_images[0].size())
batch_tensor = torch.stack(small_images)
eva_model = evaluation_model()

with open("./dataset/test.json", 'r') as f:
    test_data = json.load(f)

with open("./dataset/new_test.json", 'r') as f:
    new_test_data = json.load(f)

with open("./dataset/objects.json", 'r') as f:
    object_data = json.load(f)

test_cond_list = []
for conds in test_data:
    one_hot_array = np.zeros((24,), dtype=float)
    for cond in conds:
        one_hot_array[object_data[cond]] = 1
    test_cond_list.append(one_hot_array)

test_cond_list = torch.from_numpy(np.array(test_cond_list))

new_test_cond_list = []
for conds in new_test_data:
    one_hot_array = np.zeros((24,), dtype=float)
    for cond in conds:
        one_hot_array[object_data[cond]] = 1
    new_test_cond_list.append(one_hot_array)

new_test_cond_list = torch.from_numpy(np.array(new_test_cond_list))


# acu = eva_model.eval(batch_tensor.to("cuda:0"), test_cond_list)
# print(acu)

acu = eva_model.eval(batch_tensor.to("cuda:0"), new_test_cond_list)
print(acu)
