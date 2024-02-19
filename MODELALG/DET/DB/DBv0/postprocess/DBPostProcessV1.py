import numpy as np
import cv2
from pyclipper import PyclipperOffset
import pyclipper
from shapely.geometry import Polygon


class DBPostProcessV1:
    def __init__(self, thresh=0.3, unclip_ratio=1.5, box_thresh=0.6):
        self.min_size = 3
        self.thresh = thresh
        self.bbox_scale_ratio = unclip_ratio
        self.shortest_length = 5

    def __call__(self, _predict_score, _ori_img_shape):
        instance_score = _predict_score.squeeze()
        h, w = instance_score.shape[:2]
        height, width = _ori_img_shape[0]
        available_region = np.zeros_like(instance_score, dtype=np.float32)
        np.putmask(available_region, instance_score > self.thresh, instance_score)
        to_return_boxes = []
        to_return_scores = []
        mask_region = (available_region > 0).astype(np.uint8) * 255
        structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        refined_mask_region = cv2.morphologyEx(
            mask_region, cv2.MORPH_CLOSE, structure_element
        )
        if cv2.__version__.startswith("3"):
            _, contours, _ = cv2.findContours(
                refined_mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        elif cv2.__version__.startswith("4"):
            contours, _ = cv2.findContours(
                refined_mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            raise NotImplementedError(f"opencv {cv2.__version__} not support")
        tmp_points = []
        tmp_socre = []
        for m_contour in contours:
            if len(m_contour) < 4 and cv2.contourArea(m_contour) < 16:
                continue

            rotated_points = get_min_area_polygon(
                refined_mask_region, m_contour, self.bbox_scale_ratio
            )
            if rotated_points is None:
                continue
            if len(rotated_points) < 4:
                continue
            rotated_points = np.squeeze(rotated_points, axis=1)

            x_min_index = np.argmin(np.sum(rotated_points, axis=-1))
            points_num = len(rotated_points)
            rotated_points = np.array(
                [
                    rotated_points[(x_min_index + i) % points_num]
                    for i in range(points_num)
                ]
            )
            rotated_points[:, 0] = rotated_points[:, 0] * (width / w)
            rotated_points[:, 1] = rotated_points[:, 1] * (height / h)
            rotated_points = np.array(rotated_points, dtype=int)

            tmp_points.append(rotated_points)

            m_available_mask = np.zeros_like(available_region, dtype=np.uint8)
            cv2.drawContours(
                m_available_mask,
                [
                    m_contour,
                ],
                0,
                255,
                thickness=-1,
            )
            m_region_mask = cv2.bitwise_and(
                available_region, available_region, mask=m_available_mask
            )
            m_mask_count = np.count_nonzero(m_available_mask)
            tmp_socre.append(float(np.sum(m_region_mask) / m_mask_count))

        to_return_boxes.append(tmp_points)
        to_return_scores.append(tmp_socre)

        return to_return_boxes, to_return_scores


def get_min_area_polygon(_image, _contour, _scale_ratio=1.0):
    if abs(_scale_ratio - 1) > 0.001:
        reshaped_contour = _contour.reshape(-1, 2)
        current_polygon = Polygon(reshaped_contour)
        distance = current_polygon.area * _scale_ratio / current_polygon.length
        offset = PyclipperOffset()
        offset.AddPath(reshaped_contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        box = offset.Execute(distance)
        if len(box) == 0 or len(box) > 1:
            return None
        scaled_contour = np.array(box).reshape(-1, 1, 2)
    else:
        scaled_contour = _contour
    try:
        # rotated_box = cv2.approxPolyDP(scaled_contour, 0.001 * cv2.arcLength(scaled_contour, True), True)
        rotated_box = scaled_contour
    except Exception:
        return None
    return rotated_box
