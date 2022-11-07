import SimpleITK as sitk
import warnings
import threading


class MultiThreadIO(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

        self.output = None

    def run(self):
        self.output = self.func(*self.args)

    def get_result(self):
        if self.output is not None:
            return self.output


def read_from_DICOM_dir(DICOM_dir, dtype=sitk.sitkInt16):
    reader = sitk.ImageSeriesReader()
    list_series_ids = reader.GetGDCMSeriesIDs(DICOM_dir)

    sum_series = len(list_series_ids)
    if sum_series > 1:
        warnings.warn('Multiple series ids in this dir, only read one series')

    series_uid = list_series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(DICOM_dir, series_uid)
    image_nii = sitk.ReadImage(file_names, dtype)

    return image_nii

    # list_output = []
    # for series_i in range(sum_series):
    #     series_uid = list_series_ids[series_i]
    #     file_names = reader.GetGDCMSeriesFileNames(DICOM_dir, series_uid)
    #     image_nii = sitk.ReadImage(file_names, dtype)
    #     list_output.append(image_nii)
    # return list_output
