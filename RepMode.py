
import torch
from torch.nn import functional as F
import math
import torch.nn as nn

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
        
        #out proc
        self.c2d = nn.Sequential(nn.InstanceNorm2d(1, affine=True),
                                 nn.Mish(inplace=True),
                                 nn.Conv2d(1, 1, 3,stride=1,padding=1,padding_mode='reflect'),
                                 #RRDB(nf=1,gc=128),
                                 #RRDB(nf=32,gc=128),
                                 #RRDB(nf=32,gc=128),
                                 #nn.InstanceNorm2d(32),
                                 #nn.Mish(inplace=True),
                                 #nn.Conv2d(32, 1, 3,stride=1,padding=1,padding_mode='reflect'),
                                 )
        self.c3d64 = nn.Sequential(nn.Conv3d(256, 128, 3, stride=1,padding=1,padding_mode='reflect'),
                                 #nn.InstanceNorm3d(128, affine=True),
                                 #nn.Mish(inplace=True),
                                 nn.Conv3d(128, 1, 3, stride=1,padding=1,padding_mode='reflect'),)
        
        # self.c1d = nn.Sequential(nn.InstanceNorm2d(64, affine=True),
        #                          nn.Mish(inplace=True),
        #                          nn.Conv2d(64, 64, 3,stride=1,padding=1,padding_mode='reflect'),
        #                          RRDB(64,128))
        
        self.c1d = nn.Sequential(nn.InstanceNorm2d(64, affine=True),
                                  nn.Mish(inplace=True),
                                  nn.Conv2d(64, 32, 3,stride=1,padding=1,padding_mode='reflect'),
                                  RRDB(32,256),
                                  nn.InstanceNorm2d(32, affine=True),
                                  nn.Mish(inplace=True),
                                  nn.Conv2d(32, 64, 3,stride=1,padding=1,padding_mode='reflect'),
                                  RRDB(64,128),)
        
        self.c128 = nn.AvgPool2d(2)
        
        self.rrdb = RRDB_3D(32, 128)
        
        self.optimus = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8, activation='gelu',batch_first=True),
                                             num_layers=2)
        
        self.prime = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4096, nhead=8, activation='gelu',batch_first=True),
                                           num_layers=2)
        
        self.bumble = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4096, nhead=8, activation='gelu',batch_first=True),
                                             num_layers=2)

    def one_hot_task_embedding(self, task_id):
        N = task_id.shape[0]
        task_embedding = torch.zeros((N, self.num_tasks))
        for i in range(N):
            task_embedding[i, task_id[i]] = 1
        return task_embedding.to(self.device)

    def forward(self, x, t):
        # task embedding
        task_emb = self.one_hot_task_embedding(t)
        
        x = self.bumble(x.reshape([-1,1,64,4096]).squeeze(1)).reshape([-1,64,64,64]).unsqueeze(1)
        
        # encoding [b,1,64,64,64]
        x, x_skip1 = self.encoder_block1(x, task_emb) #[b,16,32,32,32]
        x, x_skip2 = self.encoder_block2(x, task_emb) #[b,32,16,16,16]
        #x = self.rrdb(x)
        x, x_skip3 = self.encoder_block3(x, task_emb) #[b,64,8,8,8]
        x, x_skip4 = self.encoder_block4(x, task_emb) #[b,128,4,4,4]


        # bottle
        xb = self.bottle_block(x, task_emb) #[batch,256,4,4,4]
        #x = self.prime(xb.reshape([-1,256,64])).reshape([-1,256,4,4,4])

        # decoding
        x = self.decoder_block4(xb, x_skip4, task_emb) #[b,128,8,8,8]
        x = self.decoder_block3(x, x_skip3, task_emb) #[b,64,16,16,16]
        x = self.decoder_block2(x, x_skip2, task_emb) #[b,32,32,32,32]
        #x = self.rrdb(x)
        x = self.decoder_block1(x, x_skip1, task_emb) #[b,16,64,64,64]
        x = self.conv_out(x, task_emb) #[b,1,64,64,64]
        
        x = self.c1d(x.squeeze(1)).unsqueeze(1) #[b,1,64,64,64]
        
        #Collapse the z axis
        xb = self.c3d64(xb) #[batch,1,4,4,4] || [b,1,4,8,8]
        #xb = self.c128(xb.squeeze(1)).unsqueeze(1) #[b,1,4,4,4]
        
        xb = torch.reshape(xb,[-1,1,xb.shape[-1]**3]) #[b,1,64]
        xb = self.optimus(xb)
        xb = xb.squeeze(1) 
        
        x = x.permute([0,1,3,4,2]) #move z axis last
        temp = []
        for i,s in enumerate(xb): #[8,64]
            temp.append(torch.matmul(x[i,0,:,:,:],s)) #[64,64,64] x [64] = [64,64]
        x = torch.stack(temp).unsqueeze(1)
        
        x = self.c2d(x)
        
        x = self.prime(x.reshape([-1,1,4096]).squeeze(1)).reshape([-1,64,64]).unsqueeze(1)
        
        #return outputs#F.sigmoid(outputs)
        return x.clip(0,1)


class MoDEEncoderBlock(torch.nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan)
        self.conv_down = torch.nn.Sequential(
            torch.nn.Conv3d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_chan, affine=True),#torch.nn.BatchNorm3d(out_chan),
            torch.nn.Mish(inplace=True),
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
            nn.InstanceNorm3d(out_chan, affine=True),#torch.nn.BatchNorm3d(out_chan),
            torch.nn.Mish(inplace=True),
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
        x = self.conv1(x, t) #[b,16,64,64,64] #[...,32,32,32] ... 4,4,4
        #x = self.rrdb(x)
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
                nn.InstanceNorm3d(out_chan, affine=True), #torch.nn.BatchNorm3d(out_chan),
                torch.nn.Mish(inplace=True),
            )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)


    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode='fan_out')
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

        g = self.gate(t)
        g = g.view((N, self.num_experts, self.out_chan))
        g = self.softmax(g)

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

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
    
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
    
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
    
class ResidualDenseBlock_5C_3D(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_3D, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv3d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB_3D(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_3D, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_3D(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_3D(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_3D(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x