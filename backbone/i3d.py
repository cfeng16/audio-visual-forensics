# modified from https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
import torch
import torch.nn as nn

# tiny differences with original tensorflow s3d:
# 1. torch.nn.BatchNorm3d:
#    we use: pytorch default - torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1)
#    tensorflow version: torch.nn.BatchNorm3d(num_features, eps=1e-3, momentum=0.001) -- effect: running stat updates slower
# 2. initialization:
#    we use: pytorch default - torch.nn.init.kaiming_normal_(mode='fan_in', nonlinearity='leaky_relu')
#    tensorflow version: equivalent to pytorch kaiming_normal(mode='fan_in', nonlinearity='leaky_relu'),
#                        but truncated within 2 std

class Unit3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # init
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
            
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        assert isinstance(out_channels, list)
        assert len(out_channels) == 6

        [num_out_0_0a, 
        num_out_1_0a, num_out_1_0b,
        num_out_2_0a, num_out_2_0b, 
        num_out_3_0b] = out_channels

        self.b0 = nn.Sequential(
            Unit3D(in_channels=in_channels, out_channels=num_out_0_0a, kernel_size=1, stride=1, padding=0)
            )
        self.b1 = nn.Sequential(
            Unit3D(in_channels=in_channels, out_channels=num_out_1_0a, kernel_size=1, stride=1, padding=0),
            Unit3D(in_channels=num_out_1_0a, out_channels=num_out_1_0b, kernel_size=3, stride=1, padding=1),
            )
        self.b2 = nn.Sequential(
            Unit3D(in_channels=in_channels, out_channels=num_out_2_0a, kernel_size=1, stride=1, padding=0),
            Unit3D(in_channels=num_out_2_0a, out_channels=num_out_2_0b, kernel_size=3, stride=1, padding=1),
            )
        self.b3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Unit3D(in_channels=in_channels, out_channels=num_out_3_0b, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):    
        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        # 'MaxPool3d_2a_3x3',
        # 'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        # 'MaxPool3d_3a_3x3',
        # 'Mixed_3b',
        'Mixed_3c',
        # 'MaxPool3d_4a_3x3',
        # 'Mixed_4b',
        # 'Mixed_4c',
        # 'Mixed_4d',
        # 'Mixed_4e',
        'Mixed_4f',
        # 'MaxPool3d_5a_2x2',
        # 'Mixed_5b',
        'Mixed_5c',
        # 'Logits',
        # 'Predictions',
    )

    def __init__(self, final_endpoint='Mixed_5c', first_channel=3):
        """Initializes I3D model instance.
        Args:
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()

        self._final_endpoint = final_endpoint
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        ####################################################

        self.Conv_1a = Unit3D(first_channel, 64, kernel_size=7, stride=2, padding=3)
        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)
        if self._final_endpoint == 'Conv3d_1a_7x7': return

        ####################################################

        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Conv_2b = Unit3D(64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = Unit3D(64, 192, kernel_size=3, stride=1, padding=1) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a,  # (64, 32, 56, 56)
            self.Conv_2b,     # (64, 32, 56, 56)
            self.Conv_2c)     # (192, 32, 56, 56)
        if self._final_endpoint == 'Conv3d_2c_3x3': return

        ####################################################

        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = InceptionModule(in_channels=192, out_channels=[64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(in_channels=256, out_channels=[128, 128, 192, 32, 96, 64])

        self.block3 = nn.Sequential(
            self.MaxPool_3a,  # (192, 32, 28, 28)
            self.Mixed_3b,    # (256, 32, 28, 28)
            self.Mixed_3c)    # (480, 32, 28, 28)
        if self._final_endpoint == 'Mixed_3c': return

        ####################################################

        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = InceptionModule(in_channels=480, out_channels=[192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(in_channels=512, out_channels=[160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(in_channels=512, out_channels=[128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(in_channels=512, out_channels=[112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(in_channels=528, out_channels=[256, 160, 320, 32, 128, 128])

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)
        if self._final_endpoint == 'Mixed_4f': return

        ####################################################

        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = InceptionModule(in_channels=832, out_channels=[256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(in_channels=832, out_channels=[384, 192, 384, 48, 128, 128])

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)
        if self._final_endpoint == 'Mixed_5c': return


    def forward(self, x):
        x = self.block1(x)
        if self._final_endpoint == 'Conv3d_1a_7x7': return x
        x = self.block2(x)
        if self._final_endpoint == 'Conv3d_2c_3x3': return x
        x = self.block3(x)
        if self._final_endpoint == 'Mixed_3c': return x
        x = self.block4(x)
        if self._final_endpoint == 'Mixed_4f': return x 
        x = self.block5(x)
        return x


if __name__=='__main__':
    i3d = InceptionI3d()
