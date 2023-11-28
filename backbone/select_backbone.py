from .resnet_2d3d import * 
from .s3dg import S3D
from .i3d import InceptionI3d

def select_backbone(network, first_channel=3):
    param = {'feature_size': 1024}

    # 2d3d resnet family
    if network == 'r18':
        param['feature_size'] = 512
        model = resnet18_2d3d(first_channel=first_channel)
    elif network == 'r34':
        param['feature_size'] = 512 
        model = resnet34_2d3d(first_channel=first_channel)
    elif network == 'r50':
        model = resnet50_2d3d(first_channel=first_channel)

    # 3d resnet family
    elif network == 'r3d18':
        param['feature_size'] = 512
        model = resnet18_3d(first_channel=first_channel)
    elif network == 'r3d34':
        param['feature_size'] = 512
        model = resnet34_3d(first_channel=first_channel)
    elif network == 'r3d50':
        model = resnet50_3d(first_channel=first_channel)

    # inception family
    elif network == 'i3d':
        model = InceptionI3d(first_channel=first_channel)
    elif network == 's3d':
        model = S3D(first_channel=first_channel)
    elif network == 's3dg':
        model = S3D(first_channel=first_channel, gating=True)

    else: 
        raise NotImplementedError

    return model, param