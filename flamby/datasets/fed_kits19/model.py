from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn


class Baseline(Generic_UNet):
    def __init__(self):
        pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        conv_kernel_sizes = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
        super(Baseline, self).__init__(
            input_channels=1,
            base_num_features=32,
            num_classes=3,
            num_pool=5,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0, "inplace": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=True,
            convolutional_upsampling=True,
            max_num_features=None,
            basic_block=ConvDropoutNormNonlin,
        )
