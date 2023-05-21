# import torch
# import torchvision.transforms.functional as TF
# import torch.nn.functional as F

# import os

# from collections import OrderedDict
# from natsort import natsorted
# from glob import glob
# import cv2
# import argparse
# from model.GLAN import GLAN
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot  as  plt
# weights = '/data/Data/checkpoints/SCTNet_LOL_dataset_best/models/model_bestPSNR.pth'
# def load_checkpoint(model, weights):
#     checkpoint = torch.load(weights)
#     try:
#         model.load_state_dict(checkpoint["state_dict"])
#     except:
#         state_dict = checkpoint["state_dict"]
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:]  # remove `module.`
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)

# model = GLAN()
# model.cuda()
# load_checkpoint(model,weights=weights)
# path = '/data/Data/dataset/lol_dataset/Real_world/train/low/776.png'
# target_path = '/data/Data/GLAN/vis'
# img = Image.open(path).convert('RGB')
# input_ = TF.to_tensor(img).unsqueeze(0).cuda()
#     # input_ = TF.to_tensor(img).unsqueeze(0).cuda()
# with torch.no_grad():
#     restored,out_cnn,out_trans = model(input_)
#     out_cnn = out_cnn.permute(0, 2, 3, 1).cuda().data.cpu().numpy()
#     test1=plt.imshow(out_cnn)
#     cv2.imwrite(os.path.join(target_path, +'test.png' ), cv2.cvtColor(np.squeeze(out_cnn[1,:,:])*255,cv2.COLOR_RGB2BGR))





import os
import numpy as np
import torch
from matplotlib import pyplot as plt

def visualize_feature_map(path,name,img_batch):
    img_batch = img_batch.cuda().data.cpu().numpy()
    feature_map = np.squeeze(img_batch, axis=0)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[0]
    # row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
      feature_map_split = feature_map[i, :,:]
    #   print(feature_map_split.shape)
      feature_map_combination.append(feature_map_split)
    # plt.subplot(row, col, i + 1)
      # plt.imshow(feature_map_split)
      # plt.axis('off')
    #   plt.title('feature_map_{}'.format(0))
    target_path=os.path.join(path,name)
    if not os.path.exists(target_path):
        os.makedirs(target_path,exist_ok=True)
    
    
    feature_map_sum = sum(ele for ele in feature_map_combination)     
    
    plt.imshow(feature_map_sum) 
    plt.imsave(os.path.join(target_path,'{}.png'.format(str(name))),feature_map_sum)
    print('图片存储完成')
 
    # plt.show()
 
    # 各个特征图按1：1 叠加
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # plt.imshow(feature_map_sum)
# import os
# import numpy as np
# import torch
# from matplotlib import pyplot as plt


# def get_row_col(num_pic):
#     squr = num_pic ** 0.5
#     row = round(squr)
#     col = row + 1 if squr - row > 0 else row
#     return row, col

# def visualize_feature_map(path,name,img_batch):
#     img_batch = img_batch.cuda().data.cpu().numpy()
#     feature_map = np.squeeze(img_batch, axis=0)
 
#     feature_map_combination = []
    
    
 
#     num_pic = feature_map.shape[0]
#     row, col = get_row_col(num_pic)
#     _,axs = plt.subplots(row,col)
#     plt.margins(0,0)
#     for i,ax in enumerate(axs.flat):
#     # for i in range(0, num_pic):
#       feature_map_split = feature_map[i, :,:]
#     #   print(feature_map_split.shape)
#       feature_map_combination.append(feature_map_split)
#       # ax=plt.subplot(row, col, i + 1)
#       ax.imshow(feature_map_split,interpolation="nearest")
#       ax.axis('off')
#     # plt.tight_layout(rect=[0,0,0,0])
#       plt.subplots_adjust(left=0.75, bottom=0.2, right=1, top=0.6,hspace=0.02, wspace=0.01)
#     #   plt.title('feature_map_{}'.format(0))
#     target_path=os.path.join(path,name)
#     if not os.path.exists(target_path):
#         os.makedirs(target_path,exist_ok=True)
#     # feature_map_sum = sum(ele for ele in feature_map_combination)
#     # plt.imshow(feature_map_sum)
    
#     plt.savefig(os.path.join(target_path,'{}.png'.format(str(name))),dpi=1000)
#     # plt.imsave(os.path.join(target_path,'{}.png'.format(str(name))),feature_map_combination)
#     print('图片存储完成')

# # img = torch.rand((1,64,256,256))
# visualize_feature_map(path='/data/Data/GLAN/vis',name='demo',img_batch=img)