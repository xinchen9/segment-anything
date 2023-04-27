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

import os
import sys
import numpy as np
import cv2

import argparse
import torch
import torch.nn as nn
import torch.optim as optim


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def getlist(dir=dir):
    '''
    Obtain png images list
    parameters:
    input: string file directory /train/folder/*png folder structure
    output: list: file list 
    '''
    listname =[]
    subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
    
    for folder in subfolders:
        for file in os.listdir(folder):
        # filelist = [file for file in os.listdir(folder) if file.endswith('.png')]
            if file.endswith('.jpg'):
                file_path = os.path.join(folder, file)
                listname.append(file_path)
        
    return listname

parser = argparse.ArgumentParser(description='Fine tuning SAM model')
parser.add_argument('--batch_size', default=24,
                    type=int, help='num of batch size')
parser.add_argument('--epochs',  metavar='epochs', type=int,
                    default=300, help='training epoches')
parser.add_argument('--bilinear', action='store_true',
                    default=False, help='Use bilinear upsampling')
parser.add_argument("--num_classes", default=1)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--data_dir", type=str,
                    default="/home/chenxin4/dataset/mgh_dataset/colon")
parser.add_argument('--model_dir', default="./weights/")
args = parser.parse_args()

def main():
    print("Start fine tuning")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    val_data_dir = os.path.join(data_dir, 'val')

    train_data_dir = os.path.join(train_data_dir, "images")
    train_files_list = getlist(dir=train_data_dir)
    # val_data_dir = os.path.join(val_data_dir, "images")
    # val_files_list = getlist(dir=val_data_dir)
    print(len(train_files_list))
    length = len(train_files_list)

    MODEL_PATH = "/home/chenxin4/data/sam_models"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    model_checkpoint = os.path.join(MODEL_PATH,sam_checkpoint)
    
    sam_model = sam_model_registry[model_type](model_checkpoint)
    sam_model.to(device=device)
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters()) 
    loss_fn = torch.nn.MSELoss()
    for i in range(length):
        image_name = train_files_list[i]
        label_name = image_name.replace('images', 'labels')
        image = cv2.imread(image_name)
        image = cv2.resize(image, (1024,1024),interpolation = cv2.INTER_AREA)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label = label/255
        label = torch.from_numpy(label)
        label=torch.unsqueeze(label,0)
        label=torch.unsqueeze(label,0)
        label = label.to(device)
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_image = input_image.astype(np.float32)
        input_image = torch.from_numpy(input_image)
        input_image=torch.unsqueeze(input_image,0)
        input_image= input_image.permute(0,3,1,2)
        input_image = input_image.to(device)

        with torch.no_grad():
	        image_embedding = sam_model.image_encoder(input_image)
        
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            )
        
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, (512,512), (512,512)).to(device)
        from torch.nn.functional import threshold, normalize

        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)
        loss = loss_fn(binary_mask, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i %1000 == 2:
            print("step = {} and loss = {}".format(i, loss))
        if i %10000 ==2:
            torch.save(sam_model.state_dict(), "./sam_vit_h_4b8939_fine+tuning.pth")


        # import pdb
        # pdb.set_trace()



    # image = 


if __name__=="__main__":
    main()