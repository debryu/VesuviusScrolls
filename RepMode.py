
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import math
import matplotlib.pyplot as plt 
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = torch.nn.Conv3d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, groups = in_channels)
        self.bn_frelu = torch.nn.BatchNorm3d(in_channels)
        
    def forward(self, x):
        #print(x.shape)
        y = self.conv_frelu(x)
        y = self.bn_frelu(y)
        x = torch.max(x, y)
        return x
    
class Net(torch.nn.Module):

    def __init__(
        self,
        opts,
        mult_chan=32,
        in_channels=1,
        out_channels=1,
    ):
        super().__init__()
        self.opts = opts
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tasks = len(self.opts.adopted_datasets)
        self.num_experts = 5
        self.gpu_ids = [self.opts.gpu_ids] if isinstance(self.opts.gpu_ids, int) else self.opts.gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')

        # encoder
        self.encoder_block1 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels, self.in_channels * self.mult_chan)
        self.encoder_block2 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan, self.in_channels * self.mult_chan * 2)
        self.encoder_block3 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan * 4)
        self.encoder_block4 = MoDEEncoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 8)

        # bottle
        self.bottle_block = MoDESubNet2Conv(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 16)

        # decoder
        self.decoder_block4 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 16, self.in_channels * self.mult_chan * 8)
        self.decoder_block3 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 8, self.in_channels * self.mult_chan * 4)
        self.decoder_block2 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 4, self.in_channels * self.mult_chan * 2)
        self.decoder_block1 = MoDEDecoderBlock(self.num_experts, self.num_tasks, self.in_channels * self.mult_chan * 2, self.in_channels * self.mult_chan)

        # conv out
        self.conv_out = MoDEConv(self.num_experts, self.num_tasks, self.mult_chan, self.out_channels, kernel_size=5, padding='same', conv_type='final')
        self.condensing1 = torch.nn.ConvTranspose2d(64,1, kernel_size=2, stride=2)
        self.condensing2 = torch.nn.ConvTranspose2d(64,1, kernel_size=12, stride=12)
        self.condensing3 = torch.nn.ConvTranspose2d(64,1, kernel_size=24, stride=24)
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=12, stride=12)
        self.pooling3 = torch.nn.AvgPool2d(kernel_size=24, stride=24)
        self.decoding_layer = torch.nn.Conv3d(1,1, kernel_size = (15,21,21), stride = (1,1,1), padding = (0,10,10)).to(self.device)
        self.decoding_layer_macro = torch.nn.Conv3d(1,1, kernel_size = (19,63,63), stride = (1,1,1), padding = (0,31,31)).to(self.device)
        self.normalization = torch.nn.BatchNorm2d(1)
        self.final_activation = torch.nn.Sigmoid()
        self.logit = nn.Conv2d(64, 1, 1, 1, 0)
        self.dropout_latent = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.05)
        #self.condensing2 = torch.nn.Conv2d(64*4,1, kernel_size=25, stride=1)

    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, self.num_tasks))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(self.device)

    def forward(self, x, t):
        # task embedding
        task_emb = self.one_hot_task_embedding(t)

        # encoding
        #x = self.dropout1(x)
        print(x.shape)
        x, x_skip1 = self.encoder_block1(x, task_emb)
        x, x_skip2 = self.encoder_block2(x, task_emb)
        #x = self.dropout(x)
        x, x_skip3 = self.encoder_block3(x, task_emb)
        x, x_skip4 = self.encoder_block4(x, task_emb)
        #print(x.shape,x_skip4.shape)
        #x = self.dropout(x)
        # bottle
        x = self.bottle_block(x, task_emb)

        # decoding
        x = self.dropout_latent(x)
        x = self.decoder_block4(x, x_skip4, task_emb)
        x = self.decoder_block3(x, x_skip3, task_emb)
        #x = self.dropout(x)
        x = self.decoder_block2(x, x_skip2, task_emb)
        x = self.decoder_block1(x, x_skip1, task_emb)
        outputs = self.conv_out(x, task_emb)
        #outputs = self.dropout2(outputs)
        #print(outputs.shape)
        # Squeeze to 2D
        outputs = outputs.squeeze(1)
        ThreeDlatent = outputs
        #outputs = torch.mean(outputs, dim=1).unsqueeze(1)
         # [N, 64, 64, 64] 
        outputs = self.logit(outputs)
        # First reshape the output to [N, 64, 64, 64] -> [N, 64, 64*64]
        #in_pos_encoding = outputs.reshape(outputs.shape[0], 64, 64*64)
        # Use positional encoding instead
        #positional_encoding = PositionalEncodingPermute1D(in_pos_encoding.shape[1])
        # Dim 1 cause dim0 is the batch, we want to collapse Z which is dim1
        #weights = positional_encoding(in_pos_encoding)
        '''
        #decoding_layer = torch.nn.Conv3d(1,1, kernel_size = (15,21,21), stride = (1,1,1), padding = (0,10,10)).to(self.device)
        outputs = self.decoding_layer(outputs.unsqueeze(1))
        # Remove the first dimension
        outputs = outputs
        # Sum all the Z layers
        outputs = torch.sum(outputs, dim=2)
        # Normalize the output
        outputs = self.normalization(outputs)
        # Shape [N, 1, 64, 64]
        outputs = self.final_activation(outputs)
        '''

        '''
        outputs_scale_1 = self.condensing1(outputs)
        # [N, 64, 64, 64] -> [N, 1, 64, 64]
        outputs_scale_1 = self.pooling1(outputs_scale_1)

        outputs_scale_2 = self.condensing2(outputs)
        # [N, 64, 64, 64] -> [N, 1, 64, 64]
        outputs_scale_2 = self.pooling2(outputs_scale_2)

        outputs_scale_3 = self.condensing3(outputs)
        # [N, 64, 64, 64] -> [N, 1, 64, 64]
        outputs_scale_3 = self.pooling3(outputs_scale_3)

        outputs = torch.mean(torch.stack([outputs_scale_1, outputs_scale_2, outputs_scale_3], dim=1), 1)
        
        outputs = self.final_activation(outputs)
        '''
        #print(outputs.shape)
        #outputs = outputs.squeeze(-1).squeeze(-1).reshape(outputs.shape[0],1,64,64)
        #print(outputs.shape)
        
        return outputs, ThreeDlatent


class MoDEEncoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv3d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_chan),
            FReLU(out_chan),
            #torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip


class MoDEDecoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm3d(out_chan),
            FReLU(out_chan),
            #torch.nn.ReLU(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)

    def forward(self, x, x_skip, t):
        x = self.convt(x)
        x_cat = torch.cat((x_skip, x), 1)  # concatenate
        x_cat = self.conv_less(x_cat, t)
        return x_cat


class MoDESubNet2Conv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out):
        super().__init__()
        self.conv1 = MoDEConv(num_experts, num_tasks, n_in, n_out, kernel_size=5, padding='same')
        self.conv2 = MoDEConv(num_experts, num_tasks, n_out, n_out, kernel_size=5, padding='same')

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x


class MoDEConv(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal'):
        super().__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                torch.nn.BatchNorm3d(out_chan),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight

    def gen_avgpool_kernel(self, K):
        weight = torch.ones(K, K, K).mul(1.0 / K ** 3)
        return weight

    def trans_kernel(self, kernel, target_size):
        Dp = (target_size - kernel.shape[2]) // 2
        Hp = (target_size - kernel.shape[3]) // 2
        Wp = (target_size - kernel.shape[4]) // 2
        return F.pad(kernel, [Wp, Wp, Hp, Hp, Dp, Dp])

    def routing(self, g, N):

        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum('oidhw,dhw->oidhw', self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum('oidhw,dhw->oidhw', self.expert_avg5x5_conv, self.expert_avg5x5_pool)
        
        weights = list()
        for n in range(N):
            weight_nth_sample = torch.einsum('oidhw,o->oidhw', expert_conv5x5, g[n, 0, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_conv3x3, g[n, 1, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_conv1x1, g[n, 2, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_avg3x3, g[n, 3, :]) + \
                                torch.einsum('oidhw,o->oidhw', expert_avg5x5, g[n, 4, :])
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):

        N = x.shape[0]
        #g = self.gate(t)
        #g = g.view((N, self.num_experts, self.out_chan))
        #g = self.softmax(g)
        # G dimension: [Batch, num_experts, out_chan]
        g = torch.zeros((N, self.num_experts, self.out_chan)).to(torch.device('cuda'))
        ones = torch.ones((N,self.out_chan)).to(torch.device('cuda'))
        g[:,0,:] = ones
        
        if (g.shape[2] == 1) and False:
            print('routing', g)
        w = self.routing(g, N)

        if self.training:
            y = list()
            for i in range(N):
                y.append(F.conv3d(x[i].unsqueeze(0), w[i], bias=None, stride=1, padding='same'))
            y = torch.cat(y, dim=0)
        else:
            y = F.conv3d(x, w[0], bias=None, stride=1, padding='same')

        y = self.subsequent_layer(y)

        return y
