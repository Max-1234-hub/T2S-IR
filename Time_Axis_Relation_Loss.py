# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:00:02 2022

@author: axmao2-c
"""

from __future__ import print_function

import torch
import torch.nn as nn


class IRLoss(nn.Module):
    """Inter-Channel Correlation"""
    def __init__(self):
        super(IRLoss, self).__init__()
        
    
    def forward(self, g_s, g_t):
        Acc_pairs = zip([x[0] for x in g_s], [y[0] for y in g_t])
        Gyr_pairs = zip([x[1] for x in g_s], [y[1] for y in g_t])
        Acc_loss =  [self.batch_loss(acc_f_s, acc_f_t) for acc_f_s, acc_f_t in Acc_pairs]
        Gyr_loss =  [self.batch_loss(gyr_f_s, gyr_f_t) for gyr_f_s, gyr_f_t in Gyr_pairs]
        
        total_acc_loss = (Acc_loss[0] + Acc_loss[1] + Acc_loss[2] + Acc_loss[3] + Acc_loss[4]) / 5
        total_gyr_loss = (Gyr_loss[0] + Gyr_loss[1] + Gyr_loss[2] + Gyr_loss[3] + Gyr_loss[4]) / 5
        
        total_loss = total_acc_loss + total_gyr_loss
        
        return total_loss
        
    def batch_loss(self, f_s, f_t): #original size: [bsz, c, axis, time]
        bsz, axis, time = f_s.shape[0], f_s.shape[2], f_s.shape[3]
         
        ### Time relation loss;
        f_s_time = f_s.view(bsz, -1, time)
        f_t_time = f_t.view(bsz, -1, time)
        
        ###torch.bmm()指的是矩阵之间的相乘；normalize()函数指的是向量单位化。
        emd_s_time = torch.bmm(f_s_time.permute(0,2,1), f_s_time)  # size= [batch_size*time*time]
        emd_s_time = torch.nn.functional.normalize(emd_s_time, dim = 2) # size= [batch_size*time*time]

        emd_t_time = torch.bmm(f_t_time.permute(0,2,1), f_t_time)  # size= [batch_size*time*time]
        emd_t_time = torch.nn.functional.normalize(emd_t_time, dim = 2) # size= [batch_size*time*time]
        
        cos_simil_time = torch.cosine_similarity(emd_s_time, emd_t_time, dim=2)
        loss_time = cos_simil_time.sum() / (time*bsz)
        
        ### Axis relation loss;
        f_s_axis = f_s.permute(0,2,1,3)  #size: [bsz, axis, c, time]
        f_t_axis = f_t.permute(0,2,1,3)
        
        new_f_s_axis = f_s_axis.reshape(bsz, axis, -1) #size: [bsz, axis, c*time]
        new_f_t_axis = f_t_axis.reshape(bsz, axis, -1)
        
        ###torch.bmm()指的是矩阵之间的相乘；normalize()函数指的是向量单位化。
        emd_s_axis = torch.bmm(new_f_s_axis, new_f_s_axis.permute(0,2,1))  # size= [batch_size*axis*axis]
        emd_s_axis = torch.nn.functional.normalize(emd_s_axis, dim = 2) # size= [batch_size*axis*axis]

        emd_t_axis = torch.bmm(new_f_t_axis, new_f_t_axis.permute(0,2,1))  # size= [batch_size*axis*axis]
        emd_t_axis = torch.nn.functional.normalize(emd_t_axis, dim = 2) # size= [batch_size*axis*axis]
        
        cos_simil_axis = torch.cosine_similarity(emd_s_axis, emd_t_axis, dim=2)
        loss_axis = cos_simil_axis.sum() / (axis*bsz)
        
        total_loss = (1/2)*loss_time  + (1/2)*loss_axis
        # print(total_loss)
        
        return total_loss
        

# if __name__ == "__main__":
#     kd = ICKDLoss()
#     x1 = torch.randn(2,15,224,224)
#     x2 = torch.randn(2,15,224,224)
#     kd_loss = kd([x1], [x2])
#     print(kd_loss)