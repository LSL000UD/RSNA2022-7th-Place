import SimpleITK as sitk

from Utils.CommonTools.sitk_base import resample


def sitk_dummy_3D_resample(image_nii, new_spacing, interp_xy, interp_z, out_dtype, constant_value,
                           new_origin=None,
                            new_size=None,
                            new_direction=None,
                           center_origin=None):
    """
    :param center_origin:
    :param new_origin:
    :param image_nii:
    :param new_spacing: (X, Y, Z)
    :param interp_xy: eg. sitk.Linear
    :param interp_z: eg. sitk.Linear
    :param out_dtype: eg. sitk.sitkUint8
    :param constant_value: eg. -1024
    :return:
    """
    ori_spacing = image_nii.GetSpacing()
    image_nii = resample(
        image_nii,
        new_spacing=(new_spacing[0], new_spacing[1], ori_spacing[2]),
        new_origin=None,
        new_size=None,
        new_direction=None,
        center_origin=None,
        interp=interp_xy,
        dtype=out_dtype,
        constant_value=constant_value
    )

    image_nii = resample(
        image_nii,
        new_spacing=(new_spacing[0], new_spacing[1], new_spacing[2]),
        new_origin=new_origin,
        new_size=new_size,
        new_direction=new_direction,
        center_origin=center_origin,
        interp=interp_z,
        dtype=out_dtype,
        constant_value=constant_value
    )

    return image_nii
