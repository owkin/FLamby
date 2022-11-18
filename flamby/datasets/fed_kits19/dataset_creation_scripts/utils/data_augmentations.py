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

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import (
    DataChannelSelectionTransform,
    SegChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)
from nnunet.training.data_augmentation.custom_transforms import (
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
    ConvertSegmentationToRegionsTransform,
    MaskTransform,
)
from nnunet.training.data_augmentation.pyramid_augmentations import (
    ApplyRandomBinaryOperatorTransform,
    MoveSegAsOneHotToData,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
except ImportError:
    NonDetMultiThreadedAugmenter = None


def transformations(patch_size, params, border_val_seg=-1, regions=None):
    assert (
        params.get("mirror") is None
    ), "old version of params, use new keyword do_mirror"
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(
            DataChannelSelectionTransform(params.get("selected_data_channels"))
        )

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(
            SegChannelSelectionTransform(params.get("selected_seg_channels"))
        )

    # don't do color augmentations while in 2d mode with 3d data because the
    # color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=1,
            random_crop=params.get("random_crop"),
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(
            MaskTransform(
                mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0
            )
        )

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get(
        "move_last_seg_chanel_to_data"
    ):
        tr_transforms.append(
            MoveSegAsOneHotToData(
                1, params.get("all_segmentation_labels"), "seg", "data"
            )
        )
        if (
            params.get("cascade_do_cascade_augmentations")
            and not None
            and params.get("cascade_do_cascade_augmentations")
        ):
            tr_transforms.append(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(
                        range(-len(params.get("all_segmentation_labels")), 0)
                    ),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size"),
                )
            )
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(
                        range(-len(params.get("all_segmentation_labels")), 0)
                    ),
                    key="data",
                    p_per_sample=params.get("cascade_remove_conn_comp_p"),
                    fill_with_other_class_p=params.get(
                        "cascade_remove_conn_comp_max_size_percent_threshold"
                    ),
                    dont_do_if_covers_more_than_X_percent=params.get(
                        "cascade_remove_conn_comp_fill_with_other_class_p"
                    ),
                )
            )

    if regions is not None:
        tr_transforms.append(
            ConvertSegmentationToRegionsTransform(regions, "target", "target")
        )

    tr_transforms.append(RenameTransform("seg", "target", True))
    tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
    tr_transforms = Compose(tr_transforms)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(
            DataChannelSelectionTransform(params.get("selected_data_channels"))
        )
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(
            SegChannelSelectionTransform(params.get("selected_seg_channels"))
        )

    if params.get("move_last_seg_chanel_to_data") is not None and params.get(
        "move_last_seg_chanel_to_data"
    ):
        val_transforms.append(
            MoveSegAsOneHotToData(
                1, params.get("all_segmentation_labels"), "seg", "data"
            )
        )

    val_transforms.append(RenameTransform("seg", "target", True))

    if regions is not None:
        val_transforms.append(
            ConvertSegmentationToRegionsTransform(regions, "target", "target")
        )

    val_transforms.append(NumpyToTensor(["data", "target"], "float"))
    val_transforms = Compose(val_transforms)

    return tr_transforms, val_transforms
