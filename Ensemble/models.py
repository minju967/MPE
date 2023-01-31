import torch
import torch.nn as nn
import sys
import os
import numpy as np
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50
from YOLO.yolo_call import YOLO_Model
from torchvision import transforms
from pathlib import Path
from PIL import Image
from YOLO.models.common import DetectMultiBackend
from YOLO.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from YOLO.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from YOLO.utils.plots import Annotator, colors, save_one_box
from YOLO.utils.torch_utils import select_device, smart_inference_mode

class Model_Ensemble(nn.Module):
    def __init__(self, pt_path1, pt_path2, pt_path3) -> None:
        super().__init__()

        self.model_1 = resnet34(weights=None)
        self.model_2 = resnet34(weights=None)

        self.model_1.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 5))
        self.model_2.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 2))
        
        model_1_state_dict = {}
        model_2_state_dict = {}

        model1_state_dict = torch.load(pt_path1)
        for k in list(model1_state_dict.keys()):
                model_1_state_dict[k[len("model."):]] = model1_state_dict[k]

        model2_state_dict = torch.load(pt_path3)
        for k in list(model2_state_dict.keys()):
                model_2_state_dict[k[len("model."):]] = model2_state_dict[k]
        
        log1 = self.model_1.load_state_dict(model_1_state_dict, strict=False)
        log2 = self.model_2.load_state_dict(model_2_state_dict, strict=False)
        
        print(log1)
        print(log2)

        self.model_1 = self.model_1.cuda()
        self.model_2 = self.model_2.cuda()


        self.model_3 = YOLO_Model(exp=pt_path2).YOLO  # DetectMultiBackend()
        self.stride, self.names, self.pt = self.model_3.stride, self.model_3.names, self.model_3.pt


        for model in [self.model_1, self.model_2, self.model_3]:
            for name, param in model.named_parameters():
                param.requires_grad = False

    def yolo_predict(self, batch_im, batch_path):
        imgsz =  (416, 416)
        conf_thres = 0.5
        iou_thres = 0.45
        classes = None
        agnostic_nms = False
        max_det = 1000
        line_thickness = 2 
        names = ['A', 'HSM', 'Wire']

        self.model_3.warmup(imgsz=(1 if self.pt or self.model_3.triton else im.shape[0], 3, *imgsz))  # warmup
        _, _, dt = 0, [], (Profile(), Profile(), Profile())

        result = torch.zeros(size=(1,3)).to(self.model_3.device)

        for idx in range(batch_im.shape[0]):
            path = batch_path[idx]
            im = batch_im[idx].unsqueeze(0)
            im0s = Image.open(path).convert('RGB')
            im0s = im0s.resize((416,416))
            im0 = np.ascontiguousarray(im0s)   

            # Inference
            with dt[1]:
                pred = self.model_3(im, augment=False, visualize=False)
                # pred: length 1 list
                # pred[0] type: tensor, size:(1, 10647, 8)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                '''
                    pred: length 1 list
                    pred[0] = tensor, size: (number of detected instances, 6)
                    pred[0][i]: x,y,w,h,conf,cls 
                    [tensor([[184.05009,  57.47016, 238.51973, 112.97811,   0.96094,   2.00000],
                            [184.61163, 296.76382, 237.06961, 354.71915,   0.95790,   2.00000],
                            [193.13242, 186.82487, 230.43134, 224.85338,   0.93874,   2.00000]], device='cuda:0')]                
                    or
                    [tensor([], device='cuda:0', size=(0, 6))]
                '''

            output = pred[0]
            conf_dict = {0:0.0, 1:0.0, 2:0.0}
            num_det = output.size(dim=0)

            if num_det == 0:
                yolo_output = torch.zeros(3).to(self.model_3.device)
                yolo_output = yolo_output.unsqueeze(0)
                # print(yolo_output)
            else:
                for i in range(num_det):
                    c = int(output[i][5])
                    conf = output[i][4].item()
                    if conf_dict[c] < conf:
                        conf_dict[c] = conf
                val =  list(conf_dict.values())
                yolo_output =  torch.tensor(val).to(self.model_3.device)
                yolo_output = yolo_output.unsqueeze(0)
                # print(yolo_output)
            
            result = torch.cat((result, yolo_output), dim=0)

        return result[1:]

    def forward(self, x1, x2, path):
        # x1: resnet image (224, 224)
        # x2: yolo image (416,416)
        num5_output = self.model_1(x1)
        num2_output = self.model_2(x1)
        num3_output = self.yolo_predict(x2, path)

        ensemble_vector = torch.cat((num5_output, num2_output, num3_output), dim=1)
        return ensemble_vector