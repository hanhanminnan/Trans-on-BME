from torch.autograd import Variable
from  convolution_lstm import ConvLSTM 
import torch.nn as nn
import torch
from non_local_gaussian import NONLocalBlock3D,NONLocalBlock2D,NONLocalBlock1D
from ChannelAttention import ChannelAttention,SpatialAttention
from ChannelAttention import CAM_Module
from functools import reduce
import torch.nn.functional as F
import os


# spatial-attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # kernel_size 如果为 7 的话 padding 就为3
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# Triplet 中的卷积
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, ln1=True, bias=False):

        super(BasicConv, self).__init__()
        self.channels = 3
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation = dilation, groups = groups, bias=bias)
        self.ln1 = torch.nn.LayerNorm([self.channels, 8], eps=1e-05, elementwise_affine=True)
        self.ln2 = torch.nn.LayerNorm([8, self.channels], eps=1e-05, elementwise_affine=True)
        self.ln3 = torch.nn.LayerNorm([self.channels, self.channels], eps=1e-05, elementwise_affine=True)
        self.prelu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.ln1 is not None:
            if x.shape[-2:] == [self.channels, 8]:
                x = self.ln1(x)
            if x.shape[-2:] == [8, self.channels]:
                x = self.ln2(x)
            if x.shape[-2:] == [self.channels, self.channels]:
                x = self.ln3(x)
        if self.prelu is not None:
            x = self.prelu(x)
        return x


# Z-pool 操作
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# 旋转后进行conv sigmoid操作
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


# 对C*H C*W进行select
class SKConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tensor1, tensor2):
        U_out = []
        U_out.append(tensor1)
        U_out.append(tensor2)
        batch_size = tensor1.size(0)
        U = reduce(lambda x, y: x + y, U_out)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, U_out, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


# TripletAttention操作.
class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()

        self.channels = 3
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.SKConv = SKConv()
        self.SpatialGate = SpatialGate()
        self.no_spatial = False
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out11 = self.conv1(x_out11)
        x_out21 = self.conv2(x_out21)
        V = self.SKConv(x_out11, x_out21)
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = self.conv1(x_out)
            x_out = (1 / 2) * (x_out + V)
        else:
            x_out = V
        return x_out


# channel attention module:
class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()

        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        # view()将tensor维度变为指定维度，-1表示剩下的值一起构成一个维度
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # permute()做维度换位
        energy = torch.bmm(proj_query, proj_key)
        # torch.bmm()做矩阵乘法
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        # print("2", energy_new)
        # 这句话防止梯度爆炸
        # expand_as()把一个tensor变成和函数括号内一样形状的tensor
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height)
        out = self.gamma * out + x
        return out


# ND-Non-local-block
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# 1D - NonLocalBlock
class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# 2D - NonLocalBlock
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# 3D - NonLocalBlock
class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# 最终的总Module
class TripletHACDueling3agent(nn.Module):
    def __init__(self, agents, frame_history, number_actions=6, xavier=True):
        super(TripletHACDueling3agent, self).__init__()

        self.number_actions = number_actions
        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 前面的三层卷积部分 conv0 ~ prelu3
        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(self.device)

        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)

        self.prelu0 = nn.PReLU().to(self.device)

        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(self.device)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)

        self.prelu1 = nn.PReLU().to(self.device)

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)

        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)

        self.prelu2 = nn.PReLU().to(self.device)

        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(self.device)

        self.prelu3 = nn.PReLU().to(self.device)
        self.prelu4 = nn.PReLU().to(self.device)
        self.prelu5 = nn.PReLU().to(self.device)
        self.prelu6 = nn.PReLU().to(self.device)
        self.prelu7 = nn.PReLU().to(self.device)

        # 这部分是state-value module FC
        self.fc1_val = nn.ModuleList(
            [nn.Linear(in_features=64 * 2, out_features=32).to(
                self.device) for _ in range(self.agents)])

        self.prelu6 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc2_val = nn.ModuleList(
            [nn.Linear(in_features=32 * 2, out_features=8).to(
                self.device) for _ in range(self.agents)])

        self.prelu7 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc3_val = nn.ModuleList(
            [nn.Linear(in_features=8 * 2, out_features=1).to(
                self.device) for _ in range(self.agents)])

        # 这部分是 action-value module FC
        self.fc1_adv = nn.ModuleList(
            [nn.Linear(in_features=64 * 2, out_features=32).to(
                self.device) for _ in range(self.agents)])

        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc2_adv = nn.ModuleList(
            [nn.Linear(in_features=32 * 2, out_features=8).to(
                self.device) for _ in range(self.agents)])

        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc3_adv = nn.ModuleList(
            [nn.Linear(in_features=8 * 2, out_features=number_actions).to(
                self.device) for _ in range(self.agents)])

        self.conv4 = nn.Conv3d(
            in_channels=64,
            out_channels=16,
            kernel_size=(1, 1, 1),
            padding=0).to(
            self.device)

        self.prelu_10 = nn.PReLU().to(self.device)

        self.conv5 = nn.Conv3d(
            in_channels=16,
            out_channels=4,
            kernel_size=(1, 1, 1),
            padding=0).to(
            self.device)
        self.prelu_11 = nn.PReLU().to(self.device)

        self.conv6 = nn.Conv3d(
            in_channels=4,
            out_channels=1,
            kernel_size=(1, 1, 1),
            padding=0
        ).to(self.device)

        self.prelu_12 = nn.PReLU().to(self.device)

        self.TripletAttention = TripletAttention()

        self.ln_first = torch.nn.LayerNorm([5, 8, 8], eps=1e-05, elementwise_affine=True)

        # channel-attention module - 3 All
        self.CAM_0 = CAM_Module(agents).to(self.device)
        self.CAM_1 = CAM_Module(agents).to(self.device)
        self.CAM_2 = CAM_Module(agents).to(self.device)

        # layernorm
        self.ln_adv1 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_adv2 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

        self.CAM_11 = CAM_Module(agents).to(self.device)
        self.CAM_22 = CAM_Module(agents).to(self.device)

        self.ln_val1 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_val2 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

        # non-local - 3
        self.nl_4 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_5 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_6 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)

        # layer-norm -3
        self.ln_00 = torch.nn.LayerNorm([agents, 64], eps=1e-05, elementwise_affine=True)

        self.ln_adv11 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_adv22 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

        # non-local -3
        self.nl_44 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_55 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_66 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)

        self.ln_0 = torch.nn.LayerNorm([agents, 128], eps=1e-05, elementwise_affine=True)

        self.ln_val11 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_val22 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

    def forward(self, input):
        """
        (batch_size, agents, frame_history, *image_size)
                     to
        (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device) / 255.0
        batch_size = input1.shape[0]
        # Shared layers
        input2 = []
        for i in range(self.agents):
            x = input1[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv4(x)
            x = self.prelu_10(x)
            x = self.conv5(x)
            x = self.prelu_11(x)
            x = self.conv6(x)
            x = self.prelu_12(x)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)
        # 在此消去一个dim
        input2 = input2.squeeze(2)
        input2 = input2.view([batch_size, -1, 8, 8])
        input2 = self.TripletAttention(input2)
        x4 = input2.size(0)
        input2 = input2.view(batch_size, -1, 64)

        # Communication layers
        comm = input2
        # 一个LAM操作 直接用的mean 就行
        comm = comm.unsqueeze(1)
        comm = self.nl_4(comm)
        comm = comm.squeeze(1)
        comm = self.ln_00(comm)
        comm = self.CAM_0(comm)
        comm = self.ln_00(comm)
        comm = torch.mean(comm, axis=1)

        input3_adv = []
        input3_val = []
        for i in range(self.agents):
            x = input2[:, i]
            x_adv = self.fc1_adv[i](torch.cat((x, comm), axis=-1))
            x_val = self.fc1_val[i](torch.cat((x, comm), axis=-1))
            input3_adv.append(self.prelu4[i](x_adv))
            input3_val.append(self.prelu6[i](x_val))
        # 到这里算LAM的一层结束.

        input3_adv = torch.stack(input3_adv, dim=1)
        input3_val = torch.stack(input3_val, dim=1)

        comm_val = input3_val
        comm_adv = input3_adv

        comm_val = comm_val.unsqueeze(1)
        comm_val = self.nl_5(comm_val)
        comm_val = comm_val.squeeze(1)
        comm_val = self.ln_val1(comm_val)
        comm_val = self.CAM_1(comm_val)
        comm_adv = comm_adv.unsqueeze(1)
        comm_adv = self.nl_55(comm_adv)
        comm_adv = comm_adv.squeeze(1)
        comm_adv = self.ln_adv1(comm_adv)
        comm_adv = self.CAM_11(comm_adv)
        comm_adv = self.ln_adv11(comm_adv)
        comm_val = self.ln_val11(comm_val)
        comm_val = torch.mean(comm_val, axis=1)
        comm_adv = torch.mean(comm_adv, axis=1)
        input4_val = []
        input4_adv = []
        for i in range(self.agents):
            x = input3_val[:, i]
            y = input3_adv[:, i]
            x_val = self.fc2_val[i](torch.cat((x, comm_val), axis=-1))
            x_adv = self.fc2_adv[i](torch.cat((y, comm_adv), axis=-1))
            input4_val.append(self.prelu5[i](x_val))
            input4_adv.append(self.prelu7[i](x_adv))

        input4_adv = torch.stack(input4_adv, dim=1)
        input4_val = torch.stack(input4_val, dim=1)

        comm_val = input4_val
        comm_adv = input4_adv
        comm_val = comm_val.unsqueeze(1)
        comm_val = self.nl_6(comm_val)
        comm_val = comm_val.squeeze(1)
        comm_val = self.ln_val2(comm_val)
        comm_val = self.CAM_2(comm_val)
        comm_adv = comm_adv.unsqueeze(1)
        comm_adv = self.nl_66(comm_adv)
        comm_adv = comm_adv.squeeze(1)
        comm_adv = self.ln_adv2(comm_adv)
        comm_adv = self.CAM_22(comm_adv)
        comm_adv = self.ln_adv22(comm_adv)
        comm_val = self.ln_val22(comm_val)
        comm_val = torch.mean(comm_val, axis=1)
        comm_adv = torch.mean(comm_adv, axis=1)

        output = []
        for i in range(self.agents):
            x = input4_val[:, i]
            y = input4_adv[:, i]
            x_val = self.fc3_val[i](torch.cat((x, comm_val), axis=-1)).expand(x4, self.number_actions)
            x_adv = self.fc3_adv[i](torch.cat((y, comm_adv), axis=-1))
            x = x_val + x_adv - x_adv.mean(1).unsqueeze(1).expand(x4, self.number_actions)
            output.append(x)
        output = torch.stack(output, dim=1)
        return output.cpu()


class TripletHACDueling5agent(nn.Module):
    def __init__(self, agents, frame_history, number_actions=6, xavier=True):
        super(TripletHACDueling5agent, self).__init__()

        self.number_actions = number_actions
        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 前面的三层卷积部分 conv0 ~ prelu3
        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(self.device)

        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)

        self.prelu0 = nn.PReLU().to(self.device)

        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(self.device)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)

        self.prelu1 = nn.PReLU().to(self.device)

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)

        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)

        self.prelu2 = nn.PReLU().to(self.device)

        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(self.device)

        self.prelu3 = nn.PReLU().to(self.device)
        self.prelu4 = nn.PReLU().to(self.device)
        self.prelu5 = nn.PReLU().to(self.device)
        self.prelu6 = nn.PReLU().to(self.device)
        self.prelu7 = nn.PReLU().to(self.device)

        # 这部分是state-value module FC
        self.fc1_val = nn.ModuleList(
            [nn.Linear(in_features=64 * 2, out_features=32).to(
                self.device) for _ in range(self.agents)])

        self.prelu6 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc2_val = nn.ModuleList(
            [nn.Linear(in_features=32 * 2, out_features=8).to(
                self.device) for _ in range(self.agents)])

        self.prelu7 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc3_val = nn.ModuleList(
            [nn.Linear(in_features=8 * 2, out_features=1).to(
                self.device) for _ in range(self.agents)])

        # 这部分是 action-value module FC
        self.fc1_adv = nn.ModuleList(
            [nn.Linear(in_features=64 * 2, out_features=32).to(
                self.device) for _ in range(self.agents)])

        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc2_adv = nn.ModuleList(
            [nn.Linear(in_features=32 * 2, out_features=8).to(
                self.device) for _ in range(self.agents)])

        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        self.fc3_adv = nn.ModuleList(
            [nn.Linear(in_features=8 * 2, out_features=number_actions).to(
                self.device) for _ in range(self.agents)])

        self.conv4 = nn.Conv3d(
            in_channels=64,
            out_channels=16,
            kernel_size=(1, 1, 1),
            padding=0).to(
            self.device)

        self.prelu_10 = nn.PReLU().to(self.device)

        self.conv5 = nn.Conv3d(
            in_channels=16,
            out_channels=4,
            kernel_size=(1, 1, 1),
            padding=0).to(
            self.device)
        self.prelu_11 = nn.PReLU().to(self.device)

        self.conv6 = nn.Conv3d(
            in_channels=4,
            out_channels=1,
            kernel_size=(1, 1, 1),
            padding=0
        ).to(self.device)

        self.prelu_12 = nn.PReLU().to(self.device)

        self.TripletAttention = TripletAttention()

        self.ln_first = torch.nn.LayerNorm([5, 8, 8], eps=1e-05, elementwise_affine=True)

        # channel-attention module - 3 All
        self.CAM_0 = CAM_Module(agents).to(self.device)
        self.CAM_1 = CAM_Module(agents).to(self.device)
        self.CAM_2 = CAM_Module(agents).to(self.device)

        # layernorm
        self.ln_adv1 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_adv2 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

        self.CAM_11 = CAM_Module(agents).to(self.device)
        self.CAM_22 = CAM_Module(agents).to(self.device)

        self.ln_val1 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_val2 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

        # non-local - 3
        self.nl_4 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_5 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_6 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)

        # layer-norm -3
        self.ln_00 = torch.nn.LayerNorm([agents, 64], eps=1e-05, elementwise_affine=True)

        self.ln_adv11 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_adv22 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

        # non-local -3
        self.nl_44 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_55 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)
        self.nl_66 = NONLocalBlock2D(1, sub_sample=True, bn_layer=True).to(self.device)

        self.ln_0 = torch.nn.LayerNorm([agents, 128], eps=1e-05, elementwise_affine=True)

        self.ln_val11 = torch.nn.LayerNorm([agents, 32], eps=1e-05, elementwise_affine=True)
        self.ln_val22 = torch.nn.LayerNorm([agents, 8], eps=1e-05, elementwise_affine=True)

    def forward(self, input):
        """
        (batch_size, agents, frame_history, *image_size)
                     to
        (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device) / 255.0
        batch_size = input1.shape[0]
        # Shared layers
        input2 = []
        for i in range(self.agents):
            x = input1[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv4(x)
            x = self.prelu_10(x)
            x = self.conv5(x)
            x = self.prelu_11(x)
            x = self.conv6(x)
            x = self.prelu_12(x)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)
        # 在此消去一个dim
        input2 = input2.squeeze(2)
        input2 = input2.view([batch_size, -1, 8, 8])
        input2 = self.TripletAttention(input2)
        x4 = input2.size(0)
        input2 = input2.view(batch_size, -1, 64)

        # Communication layers
        comm = input2
        # 一个LAM操作 直接用的mean 就行
        comm = comm.unsqueeze(1)
        comm = self.nl_4(comm)
        comm = comm.squeeze(1)
        comm = self.ln_00(comm)
        comm = self.CAM_0(comm)
        comm = self.ln_00(comm)
        comm = torch.mean(comm, axis=1)

        input3_adv = []
        input3_val = []
        for i in range(self.agents):
            x = input2[:, i]
            x_adv = self.fc1_adv[i](torch.cat((x, comm), axis=-1))
            x_val = self.fc1_val[i](torch.cat((x, comm), axis=-1))
            input3_adv.append(self.prelu4[i](x_adv))
            input3_val.append(self.prelu6[i](x_val))
        # 到这里算LAM的一层结束.

        input3_adv = torch.stack(input3_adv, dim=1)
        input3_val = torch.stack(input3_val, dim=1)

        comm_val = input3_val
        comm_adv = input3_adv

        comm_val = comm_val.unsqueeze(1)
        comm_val = self.nl_5(comm_val)
        comm_val = comm_val.squeeze(1)
        comm_val = self.ln_val1(comm_val)
        comm_val = self.CAM_1(comm_val)
        comm_adv = comm_adv.unsqueeze(1)
        comm_adv = self.nl_55(comm_adv)
        comm_adv = comm_adv.squeeze(1)
        comm_adv = self.ln_adv1(comm_adv)
        comm_adv = self.CAM_11(comm_adv)
        comm_adv = self.ln_adv11(comm_adv)
        comm_val = self.ln_val11(comm_val)
        comm_val = torch.mean(comm_val, axis=1)
        comm_adv = torch.mean(comm_adv, axis=1)
        input4_val = []
        input4_adv = []
        for i in range(self.agents):
            x = input3_val[:, i]
            y = input3_adv[:, i]
            x_val = self.fc2_val[i](torch.cat((x, comm_val), axis=-1))
            x_adv = self.fc2_adv[i](torch.cat((y, comm_adv), axis=-1))
            input4_val.append(self.prelu5[i](x_val))
            input4_adv.append(self.prelu7[i](x_adv))

        input4_adv = torch.stack(input4_adv, dim=1)
        input4_val = torch.stack(input4_val, dim=1)

        comm_val = input4_val
        comm_adv = input4_adv
        comm_val = comm_val.unsqueeze(1)
        comm_val = self.nl_6(comm_val)
        comm_val = comm_val.squeeze(1)
        comm_val = self.ln_val2(comm_val)
        comm_val = self.CAM_2(comm_val)
        comm_adv = comm_adv.unsqueeze(1)
        comm_adv = self.nl_66(comm_adv)
        comm_adv = comm_adv.squeeze(1)
        comm_adv = self.ln_adv2(comm_adv)
        comm_adv = self.CAM_22(comm_adv)
        comm_adv = self.ln_adv22(comm_adv)
        comm_val = self.ln_val22(comm_val)
        comm_val = torch.mean(comm_val, axis=1)
        comm_adv = torch.mean(comm_adv, axis=1)

        output = []
        for i in range(self.agents):
            x = input4_val[:, i]
            y = input4_adv[:, i]
            x_val = self.fc3_val[i](torch.cat((x, comm_val), axis=-1)).expand(x4, self.number_actions)
            x_adv = self.fc3_adv[i](torch.cat((y, comm_adv), axis=-1))
            x = x_val + x_adv - x_adv.mean(1).unsqueeze(1).expand(x4, self.number_actions)
            output.append(x)
        output = torch.stack(output, dim=1)
        return output.cpu()


class DQN:
    # The class initialisation function.
    def __init__(
            self,
            agents,
            frame_history,
            logger,
            number_actions=6,
            type="TripletAttentionCommNetDueling",
            load_model=None):
        self.agents = agents
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log(f"Using {self.device}")
        self.load_model = load_model
        # Create a Q-network, which predicts the q-value for a particular state
        if type == "TripletAttentionCommNetDueling":
            self.q_network = TripletAttentionCommNetDueling(
                agents,
                frame_history,
                number_actions).to(self.device)
            self.target_network = TripletAttentionCommNetDueling(
                agents,
                frame_history,
                number_actions).to(self.device)

        self.copy_to_target_network()
        # Freezes target network
        self.target_network.train(False)
        for p in self.target_network.parameters():
            p.requires_grad = False
        # Define the optimiser which is used when updating the Q-network. The
        # learning rate determines how big each gradient step is during
        # backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=4.5e-4)#1e-3
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser, step_size=50, gamma=0.5)

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, name="dqn.pt", forced=False):
        self.logger.save_model(self.q_network.state_dict(), name, forced)

    # Function that is called whenever we want to train the Q-network. Each
    # call to this function takes in a transition tuple containing the data we
    # use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):

        self.optimiser.zero_grad()

        loss = self._calculate_loss(transitions, discount_factor)

        loss.backward()

        self.optimiser.step()
        return loss.item()

    def _calculate_loss_tf(self, transitions, discount_factor):
        import tensorflow as tf
        curr_state = transitions[0]
        self.predict_value = tf.convert_to_tensor(
            self.q_network.forward(
                torch.tensor(curr_state)).view(
                -1,
                self.number_actions).detach().numpy(),
            dtype=tf.float32)  # Only works for 1 agent
        reward = tf.squeeze(
            tf.clip_by_value(
                tf.convert_to_tensor(
                    transitions[2], dtype=tf.float32), -1, 1), [1])
        next_state = transitions[3]
        action_onehot = tf.squeeze(tf.one_hot(
            transitions[1], 6, 1.0, 0.0), [1])

        pred_action_value = tf.reduce_sum(
            self.predict_value * action_onehot, 1)  # N,

        with tf.variable_scope('target'):
            targetQ_predict_value = tf.convert_to_tensor(
                self.q_network.forward(torch.tensor(
                    next_state)).view(-1, self.number_actions)
                .detach().numpy(),
                dtype=tf.float32)   # NxA

        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        target = reward + discount_factor * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(target, pred_action_value,
                                    reduction=tf.losses.Reduction.MEAN)
        with tf.Session() as _:
            print("cost", cost.eval())

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        """
        Transitions are tuple of shape
        (states, actions, rewards, next_states, dones)
        """
        curr_state = torch.tensor(transitions[0])
        next_state = torch.tensor(transitions[3])
        terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(
            torch.tensor(
                transitions[2], dtype=torch.float32), -1, 1)

        y = self.target_network.forward(next_state)

        y= y.view(-1, self.agents, self.number_actions)

        max_target_net = y.max(-1)[0]

        network_prediction = self.q_network.forward(curr_state).view(
            -1, self.agents, self.number_actions)
        network_prediction = network_prediction.view(
            -1, self.agents, self.number_actions)
        isNotOver = (torch.ones(*terminal.shape) - terminal)

        batch_labels_tensor = rewards + isNotOver * \
            (discount_factor * max_target_net.detach())

        actions = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = torch.gather(network_prediction, -1, actions).squeeze()

        return torch.nn.SmoothL1Loss()(
                batch_labels_tensor.flatten(), y_pred.flatten())
