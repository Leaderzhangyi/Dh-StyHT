import torch.nn as nn
from model.configuration import TransModuleConfig
from model.transformer_components import TransformerDecoderLayer
import torch 
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import numpy as np
# from PIL import Image
# from torchvision.utils import save_image, make_grid
from skimage.color import rgb2gray
from skimage.feature import canny
import cv2
# import time 

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def calc_mean_std(feat, eps=1e-5):
    """
        Calculate the mean and std of feature map.
    """
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """
        Mean variance normalization.
    """
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def get_edge(img):
    img_grap = rgb2gray(img)
    edge = canny(img_grap, sigma=2,mask=None).astype(np.float64)
    edge_tensor = FF.to_tensor(edge).float()
    return edge_tensor

def BGR2HSV(img, pred_img, edge, batch):
    img_hsv = []
    pre_img_hsv = []
    edge_mask_list = []
    for n in range(batch):
        numberofimg_from = n
        numberofimg = n+1
        img_ = img[numberofimg_from:numberofimg, ...]
        img_ = img_.permute(0, 2, 3, 1) * 255
        img_ = np.concatenate(img_.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT
        hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)

        pred_img_ = pred_img[numberofimg_from:numberofimg, ...]
        pred_img_ = pred_img_.permute(0, 2, 3, 1) * 255
        pred_img_ = np.concatenate(pred_img_.cpu().detach().numpy().astype(np.uint8), axis=0)  # pre
        pre_hsv = cv2.cvtColor(pred_img_, cv2.COLOR_BGR2HSV)

        edge_ = edge[numberofimg_from:numberofimg, ...]
        edge_ = edge_.permute(0, 2, 3, 1) * 255
        edge_ = np.concatenate(edge_.cpu().detach().numpy().astype(np.uint8), axis=0)  # edge

        edge_mask = FF.to_tensor(edge_).float()
        hsv = FF.to_tensor(hsv).float()
        pre_hsv = FF.to_tensor(pre_hsv).float()
        img_hsv.append(hsv)
        pre_img_hsv.append(pre_hsv)
        edge_mask_list.append(edge_mask)
    img_hsv = np.stack([img_hsv[0],img_hsv[1],img_hsv[2],img_hsv[3]],axis=0)
    pre_img_hsv = np.stack([pre_img_hsv[0],pre_img_hsv[1],pre_img_hsv[2],pre_img_hsv[3]],axis =0)
    edge_mask_list = np.stack([edge_mask_list[0],edge_mask_list[1],edge_mask_list[2],edge_mask_list[3]],axis=0)
    img_hsv = torch.from_numpy(img_hsv).float()
    pre_img_hsv = torch.from_numpy(pre_img_hsv).float()
    edge_mask_list = torch.from_numpy(edge_mask_list).float()

    return img_hsv, pre_img_hsv,edge_mask_list
   

class HierarchiesTrans(nn.Module):
  """The Transfer Module of Style Transfer via Transformer

  Taking Transformer Decoder as the transfer module.

  Args:
    config: The configuration of the transfer module
  """
  def __init__(self, config: TransModuleConfig=None):
    super(HierarchiesTrans, self).__init__()
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(
          d_model=config.d_model,
          nhead=config.nhead,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          attn_drop=config.attn_drop,
          drop=config.drop,
          drop_path=config.drop_path,
          act_layer=config.act_layer,
          norm_layer=config.norm_layer,
          norm_first=config.norm_first
          ) \
      for i in range(config.nlayer)
    ])

  def forward(self, content_feature, style_feature):
    """
    Args:
      content_feature: Content features，for producing Q sequences. Similar to tgt sequences in pytorch. (Tensor,[Batch,sequence,dim])
      style_feature : Style features，for producing K,V sequences.Similar to memory sequences in pytorch.(Tensor,[Batch,sequence,dim])

    Returns:
      Tensor with shape (Batch,sequence,dim)
    """
    for layer in self.layers:
      content_feature = layer(content_feature, style_feature)
    
    return content_feature



class MultiscaleVGGFusionDecoder(nn.Module):
  def __init__(self, d_model=768, seq_input=False):
      super(MultiscaleVGGFusionDecoder, self).__init__()
      self.d_model = d_model
      self.seq_input = seq_input
      self.decoder = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(int(self.d_model), 256, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.ReLU(),
        # Upsample Layer 4
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1, 0),
        nn.ReLU(),
        # Channel to 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1, 0),
      )
         
  def forward(self, x, input_resolution):
    if self.seq_input == True:
      B, N, C = x.size()
#       H, W = math.ceil(self.img_H//self.patch_size), math.ceil(self.img_W//self.patch_size)
      (H, W) = input_resolution
      x = x.permute(0, 2, 1).reshape(B, C, H, W)
    x = self.decoder(x)  
    return x


