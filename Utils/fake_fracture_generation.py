import random
import numpy as np
import cv2
from Utils.CommonTools.bbox import get_bbox


class FakeFractureGenerator:
    def __init__(self):

        self.ct_th_for_pick_fracture_location = -0.35
        self.filled_ct_value = -1.

        self.range_len_z = (5, 20)
        self.range_len_y = (3, 5)
        self.range_len_x = (10, 30)

        pass

    def _random_pick_a_contour_point(self, ct_2d, seg_2d):
        contours, hierarchy = cv2.findContours(np.uint8(np.logical_and(ct_2d > self.ct_th_for_pick_fracture_location,
                                                                       seg_2d > 0)), mode=cv2.RETR_LIST,
                                               method=cv2.CHAIN_APPROX_NONE)
        rand_contour = random.sample(contours, 1)[0]

        rand_index = random.randint(0, rand_contour.shape[0] - 1)
        rand_point = rand_contour[rand_index, 0]
        rand_point = rand_point[::-1]

        return rand_point

    @staticmethod
    def _generate_remove_rectangle(center_point, image_shape, len_y, len_x):
        remove_mask = np.zeros(image_shape, np.uint8)

        center_y, center_x = center_point

        remove_mask[center_y - len_y // 2:center_y + len_y // 2,
        center_x - len_x // 2:center_x + len_x // 2] = 1
        angle = random.uniform(-180, 180)
        rotation_matrix = cv2.getRotationMatrix2D((float(center_x), float(center_y)), angle, scale=1.0)
        remove_mask = cv2.warpAffine(remove_mask,
                                     rotation_matrix,
                                     (int(remove_mask.shape[1]), int(remove_mask.shape[0])),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0,
                                     flags=cv2.INTER_NEAREST)
        return remove_mask

    def __call__(self, ct, seg, gt):
        try:
            seg = np.logical_and(seg >= 1, seg <= 7)
            seg_bbox = get_bbox(seg)
            rand_z = random.randint(seg_bbox[0], seg_bbox[1])

            seg_2d = seg[rand_z, :, :]
            rand_point = self._random_pick_a_contour_point(ct[rand_z], np.uint8(seg_2d))
            rand_y, rand_x = rand_point[0], rand_point[1]

            rand_len_y = random.randint(self.range_len_y[0], self.range_len_y[1])
            rand_len_x = random.randint(self.range_len_x[0], self.range_len_x[1])
            rand_len_z = random.randint(self.range_len_z[0], self.range_len_z[1])

            remove_mask = self._generate_remove_rectangle((rand_y, rand_x), seg_2d.shape, rand_len_y, rand_len_x)
            remove_mask_extend = cv2.dilate(remove_mask, kernel=np.ones((3, 3), np.uint8), iterations=3)

            # print(rand_z)
            # print(rand_point)

            rand_extend = rand_len_z // 2
            for i in range(rand_extend):
                if rand_z - i >= 0:
                    ct[rand_z - i][np.logical_and(remove_mask > 0, seg[rand_z - i, :, :] > 0)] = self.filled_ct_value
                    gt[rand_z - i][np.logical_and(remove_mask_extend > 0, seg[rand_z - i, :, :] > 0)] = 1
                if rand_z + i <= ct.shape[0] - 1:
                    ct[rand_z + i][np.logical_and(remove_mask > 0, seg[rand_z + i, :, :] > 0)] = self.filled_ct_value
                    gt[rand_z + i][np.logical_and(remove_mask_extend > 0, seg[rand_z + i, :, :] > 0)] = 1
            return ct, gt
        except:
            return ct, gt
