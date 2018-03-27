import torch.nn as nn
import torch
from functions import ReverseLayerF

class DSN(nn.Module):
    def __init__(self, n_class,code_size,channels = 3):
        super(DSN, self).__init__()
        #input_image 32*32
        self.channels = channels
        self.private_enc_src = nn.Sequential()
        self.private_enc_tgt = nn.Sequential()
        self.shared_enc = nn.Sequential()
        self.shared_dec = nn.Sequential()
        self.classifier = nn.Sequential()

        ##PRIVATE ENCODER for source
        self.private_enc_src.add_module('pri_enc_conv1', nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5,
                                                    padding=2))
        self.private_enc_src.add_module('pri_enc_relu1', nn.ReLU())
        self.private_enc_src.add_module('pri_enc_bn1', nn.BatchNorm2d(32))
        self.private_enc_src.add_module('pri_enc_pool1', nn.MaxPool2d(kernel_size=2, stride=2))#32*16*16

        self.private_enc_src.add_module('pri_enc_conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                           padding=2))
        self.private_enc_src.add_module('pri_enc_relu2', nn.ReLU())
        self.private_enc_src.add_module('pri_enc_bn2', nn.BatchNorm2d(64))
        self.private_enc_src.add_module('pri_enc_pool2', nn.MaxPool2d(kernel_size=2, stride=2))#64*8*8
        #reshape
        self.private_fc_src = nn.Sequential(
            nn.Linear(64 * 8 * 8, code_size),
            nn.ReLU(),
            nn.BatchNorm2d(code_size)
        )

        ##PRIVATE ENCODER for target
        self.private_enc_tgt.add_module('pri_enc_conv1_1', nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5,
                                                                   padding=2))
        self.private_enc_tgt.add_module('pri_enc_relu1_1', nn.ReLU())
        self.private_enc_tgt.add_module('pri_enc_bn1_1', nn.BatchNorm2d(32))
        self.private_enc_tgt.add_module('pri_enc_pool1_1', nn.MaxPool2d(kernel_size=2, stride=2))  # 32*16*16

        self.private_enc_tgt.add_module('pri_enc_conv2_1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                   padding=2))
        self.private_enc_tgt.add_module('pri_enc_relu2_1', nn.ReLU())
        self.private_enc_tgt.add_module('pri_enc_bn2_1', nn.BatchNorm2d(64))
        self.private_enc_tgt.add_module('pri_enc_pool2_1', nn.MaxPool2d(kernel_size=2, stride=2))  # 64*8*8
        # reshape
        self.private_fc_tgt = nn.Sequential(
            nn.Linear(64 * 8 * 8, code_size),
            nn.ReLU(),
            nn.BatchNorm2d(code_size)
        )

        ##SHARED_ENCODER
        self.shared_enc.add_module('shd_enc_conv1', nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=5,
                                                           padding=2))
        self.shared_enc.add_module('shd_enc_relu1', nn.ReLU())
        self.shared_enc.add_module('shd_enc_pool1', nn.MaxPool2d(kernel_size=3, stride=2,padding=1))#32*16*16

        self.shared_enc.add_module('shd_enc_conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                                                          padding=2))
        self.shared_enc.add_module('shd_enc_relu2', nn.ReLU())
        self.shared_enc.add_module('shd_enc_pool2', nn.MaxPool2d(kernel_size=3, stride=2,padding=1))#64*8*8
        # reshape
        self.shd_enc_fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, code_size),
            nn.ReLU()
        )
        # DOMIAN CLASSIFIER
        self.domain_dis =  nn.Sequential(
            nn.Linear(code_size, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

        # LABEL CLASSIFIER
        self.classifier = nn.Sequential(
                nn.Linear(code_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, n_class)
            )

        ##SHARED_DECODER
        self.shd_dec_fc = nn.Sequential(
            nn.Linear(code_size, 300),
            nn.ReLU(),
            nn.BatchNorm2d(300)
        )
        # reshape b*3*10*10
        self.shared_dec.add_module('shd_dec_conv1', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,padding=2))#3*10*10
        self.shared_dec.add_module('shd_dec_relu1', nn.ReLU())
        self.shared_dec.add_module('shd_dec_bn1', nn.BatchNorm2d(16))

        self.shared_dec.add_module('shd_dec_conv2', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,padding=2))
        self.shared_dec.add_module('shd_dec_relu2', nn.ReLU())
        self.shared_dec.add_module('shd_dec_bn2', nn.BatchNorm2d(16))
        self.shared_dec.add_module('shd_dec_Up2', nn.Upsample([30, 30]))#32*32*32
        self.shared_dec.add_module('shd_dec_Up2_2', nn.ReplicationPad2d([1, 1, 1, 1]))

        self.shared_dec.add_module('shd_dec_conv3', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,padding=1))
        self.shared_dec.add_module('shd_dec_relu3', nn.ReLU())
        self.shared_dec.add_module('shd_dec_bn3', nn.BatchNorm2d(16))
        self.shared_dec.add_module('shd_dec_conv4', nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=3,padding=1))
        self.shared_dec.add_module('shd_dec_bn4', nn.BatchNorm2d(channels))

    def forward(self, input_data, mode, Rec_scheme ,alpha):

        if mode =='source':
            private_enc = self.private_enc_src
            pri_enc_fc = self.private_fc_src
        else:
            input_data = input_data.expand(input_data.data.shape[0], self.channels, 32, 32)
            private_enc = self.private_enc_tgt
            pri_enc_fc = self.private_fc_src

        #private encoder
        private_feat = private_enc(input_data)
        private_feat = private_feat.view(-1, 64 * 8 * 8)
        private_feat_code = pri_enc_fc(private_feat)

        #shared encoder
        shared_feat = self.shared_enc(input_data)
        shared_feat = shared_feat.view(-1, 64 * 8 * 8)
        shared_feat_code = self.shd_enc_fc(shared_feat)

        #label classifier
        pred_label = self.classifier(shared_feat_code)

        #domain classifier
        reverse_feature = ReverseLayerF.apply(shared_feat_code,alpha)
        pred_domain = self.domain_dis(reverse_feature)

        #shared decoder
        if Rec_scheme == 'private':
            rec_feat_code = private_feat_code
        if Rec_scheme == 'shared':
            rec_feat_code = shared_feat_code
        if Rec_scheme == 'all':
            rec_feat_code = private_feat_code + shared_feat_code

        feat_encode = self.shd_dec_fc(rec_feat_code)
        feat_encode = feat_encode.view(-1, 3, 10, 10)
        img_rec = self.shared_dec(feat_encode)

        return private_feat_code, shared_feat_code, pred_label, pred_domain, img_rec