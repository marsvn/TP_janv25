import torch
from torchPETADMM.models.DRUnet import NFUNetResFix, NFUNetRes2, NFDRUNet, NFUNetResMultBranch2DecoderFix
from torchPETADMM.models.DnCNN_SN import DnCNN

class MyDeepModelSN(torch.nn.Module):
    def __init__(self):
        super(MyDeepModelSN, self).__init__()
        self.model = DnCNN(channels=1, num_of_layers=8,lip = 1.0, no_bn=True, adaptive=False)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x)
    
class MyDeepModel(torch.nn.Module):
    def __init__(self):
        super(MyDeepModel, self).__init__()
        # self.model = NFUNetResFix(in_nc=1, out_nc=1, nb=1, wf= 32, depth = 3, act_mode = 'E',
        #                      downsample_mode='strideconv', upsample_mode='upconv', 
        #                      norm = 'nonorm.v1', 
        #                      keep_bias=False, repeat=False, bUseAtt=False, bUseComposition=False, 
        #                      bUseMask=False, bUseClamp=False, use_gamma=False)
        self.model = NFUNetRes2(in_nc=1, out_nc=1, nb=1, wf= 32, depth = 3, act_mode = 'E',
                             downsample_mode='strideconv', upsample_mode='upconv', 
                             norm = 'nonorm.v1', 
                             keep_bias=False, repeat=False, bUseAtt=False, bUseComposition=False, 
                             bUseMask=False, bUseClamp=False, use_gamma=False)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x)[0]
    

class MyDeepModel2(torch.nn.Module):
    def __init__(self):
        super(MyDeepModel2, self).__init__()
        self.model = NFDRUNet(in_nc=1, out_nc=1, nb=1, wf= 32, depth = 3, act_mode = 'E',
                             downsample_mode='strideconv', 
                             norm = 'nonorm.v1', 
                             keep_bias=False, repeat=False, bUseAtt=False, bUseComposition=False, 
                             bUseMask=False, bUseClamp=False, use_gamma=False)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x) #[0]
    
class MyDeepModel3(torch.nn.Module):
    def __init__(self):
        super(MyDeepModel3, self).__init__()
        self.model = NFUNetResMultBranch2DecoderFix(in_nc=1, out_nc=1, nb=1, wf= 32, depth = 3, 
                             act_mode = 'E', upsample_mode='upconv',repeat=False,
                             downsample_mode='strideconv', 
                             norm = 'nonorm.v1',  bUse_N = False)    

        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x, xmri):
        return self.model(x, xmri)
    

class MyDenoiser(torch.nn.Module):
    def __init__(self, id = 0):
        super(MyDenoiser, self).__init__()
        if id == 0:
            self.base = MyDeepModel()
        elif id == 1:
            self.base = MyDeepModelSN()
        else:
            self.base = MyDeepModel2()

    def forward(self, x):
        return self.base(x)

class MyDenoiserMRI(torch.nn.Module):
    def __init__(self):
        super(MyDenoiserMRI, self).__init__()
        self.base = MyDeepModel3()
        

    def forward(self, x, xmri):
        return self.base(x, xmri) 




class MyLightModel(torch.nn.Module):
    def __init__(self):
        super(MyLightModel, self).__init__()
        # self.model = NFUNetResFix(in_nc=1, out_nc=1, nb=1, wf= 16, depth = 1, act_mode = 'E',
        #                      downsample_mode='strideconv', upsample_mode='upconv', 
        #                      norm = 'nonorm.v1', 
        #                      keep_bias=False, repeat=False, bUseAtt=False, bUseComposition=False, 
        #                      bUseMask=False, bUseClamp=False, use_gamma=False)
        self.model = NFUNetRes2(in_nc=1, out_nc=1, nb=1, wf= 16, depth = 1, act_mode = 'E',
                             downsample_mode='strideconv', upsample_mode='upconv', 
                             norm = 'nonorm.v1', 
                             keep_bias=False, repeat=False, bUseAtt=False, bUseComposition=False, 
                             bUseMask=False, bUseClamp=False, use_gamma=False)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x)[0]
    

class MyLDenoiser(torch.nn.Module):
    def __init__(self):
        super(MyLDenoiser, self).__init__()
        self.base = MyLightModel()

    def forward(self, x):
        return self.base(x)