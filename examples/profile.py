import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

import sys
import time

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import intel_extension_for_pytorch as ipex

image = cv2.imread('./images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# sam_checkpoint = "/home/chenxin4/data/sam_models/sam_vit_h_4b8939.pth"
#sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# sam_checkpoint = "/home/chenxin4/data/sam_models/sam_vit_l_0b3195.pth"
#sam_checkpoint = "sam_vit_l_0b3195.pth"
# model_type = "vit_l"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sam_checkpoint = "/home/chenxin4/data/sam_models/sam_vit_b_01ec64.pth"
# sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
# device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# sam = ipex.optimize(sam)
total_time = 0
for num in range(100):
    start = time.time()
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    #print(len(masks))
    end = time.time()
    total_time +=(end-start)
during = total_time/100
print(during)
