# Copyright (C) 2023 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0



import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

import sys
import time

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

image = cv2.imread('./images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
start = time.time()
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
end = time.time()
during = end-start
print(during)