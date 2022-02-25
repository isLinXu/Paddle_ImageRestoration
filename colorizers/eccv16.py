import numpy as np
import paddle
import paddle.nn as nn

from .base_color import *


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2D):
        super(ECCVGenerator, self).__init__()

        model1 = [
            nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            nn.Conv2D(64, 64, kernel_size=3, stride=2, padding=1),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            norm_layer(64),
        ]

        model2 = [
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            nn.Conv2D(128, 128, kernel_size=3, stride=2, padding=1),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            norm_layer(128),
        ]

        model3 = [
            nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=2, padding=1),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            norm_layer(256),
        ]

        model4 = [
            nn.Conv2D(256, 512, kernel_size=3, stride=1, padding=1),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            norm_layer(512),
        ]

        model5 = [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            norm_layer(512),
        ]

        model6 = [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            norm_layer(512),
        ]

        model7 = [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            norm_layer(512),
        ]

        model8 = [nn.Conv2DTranspose(512, 256, kernel_size=4, stride=2, padding=1)]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
        ]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
        ]
        model8 += [
            nn.ReLU(True),
        ]

        model8 += [
            nn.Conv2D(256, 313, kernel_size=1, stride=1, padding=0),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(axis=1)
        self.model_out = nn.Conv2D(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias_attr=False
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


def eccv16(pretrained=True):
    model = ECCVGenerator()
    if pretrained:
        model.set_state_dict(paddle.load('/home/linxu/Desktop/Paddle-Colorization/colorizers/paddle_eccv16.pdparams'))
    return model
