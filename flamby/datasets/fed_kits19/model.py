#    Copyright 2020 Division of Medical Image Computing, German Cancer Research
# Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Part of this file comes from https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet
# See flamby/datasets/fed_kits19/dataset_creation_scripts/LICENSE/README.md for more
# information

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
