import torch
from dice_loss import dice_loss

def calc_loss(prediction, target, bce_weight=0.5):
       # 计算BCE Loss，使用logits作为输入，避免重复计算sigmoid
       bce = F.binary_cross_entropy_with_logits(prediction, target)
       # 计算sigmoid，将logits转换为概率
       prediction = F.sigmoid(prediction)
       # 计算Dice Loss，使用自定义的dice_loss函数
       dice = dice_loss(prediction, target)
       # 计算总的损失，根据bce_weight的值进行加权
       loss = bce * bce_weight + dice * (1 - bce_weight)
       return loss
