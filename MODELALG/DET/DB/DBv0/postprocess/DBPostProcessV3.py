import numpy as np
import cv2
from pyclipper import PyclipperOffset
import pyclipper
from shapely.geometry import Polygon
import time
from threading import Thread, Lock


class DBPostProcessV3:
    def __init__(self, thresh=0.3, unclip_ratio=1.5, box_thresh=0.6, num_works=1):
        self.min_size = 3
        self.thresh = thresh
        self.bbox_scale_ratio = unclip_ratio
        self.shortest_length = 5
        self.lock = Lock()  # lock
        self.num_th = num_works  # thread

    def __call__(self, _predict_score, _ori_img_shape):
        # print("*" * 50 + "DB后处理耗时统计开始：")
        starttime = time.time()
        instance_score = _predict_score.squeeze()
        h, w = instance_score.shape[:2]
        height, width = _ori_img_shape[0]
        available_region = np.zeros_like(instance_score, dtype=np.float32)
        np.putmask(available_region, instance_score > self.thresh, instance_score)
        to_return_boxes = []
        to_return_scores = []
        mask_region = (available_region > 0).astype(np.uint8) * 255
        # print("1：", time.time() - starttime)
        structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # print("2：", time.time() - starttime)
        refined_mask_region = cv2.morphologyEx(
            mask_region, cv2.MORPH_CLOSE, structure_element
        )
        # print("3：", time.time() - starttime)

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
        # print("4：", time.time() - starttime)

        # multi thread
        threads = []
        tmp_points = []
        tmp_socre = []
        t_len = int(len(contours) / self.num_th)
        for i in range(self.num_th - 1):
            contour = contours[i * t_len : (i + 1) * t_len]
            thread = Thread(
                target=self.thread_contour,
                args=(
                    refined_mask_region,
                    h,
                    w,
                    height,
                    width,
                    available_region,
                    contour,
                    tmp_points,
                    tmp_socre,
                ),
            )
            threads.append(thread)
            thread.start()

        # print("5：", time.time() - starttime)
        contour = contours[(self.num_th - 1) * t_len :]
        thread = Thread(
            target=self.thread_contour,
            args=(
                refined_mask_region,
                h,
                w,
                height,
                width,
                available_region,
                contour,
                tmp_points,
                tmp_socre,
            ),
        )
        # print("6：", time.time() - starttime)
        threads.append(thread)
        thread.start()
        # print("7：", time.time() - starttime)

        for t in threads:
            t.join()
        # print("8：", time.time() - starttime)

        to_return_boxes.append(tmp_points)
        to_return_scores.append(tmp_socre)
        # print("*" * 50 + "DB后处理总耗时：", time.time() - starttime)
        return to_return_boxes, to_return_scores

    # contour for
    def thread_contour(
        self,
        refined_mask_region,
        h,
        w,
        height,
        width,
        available_region,
        contours,
        tmp_points,
        tmp_socre,
    ):
        for m_contour in contours:
            if not (len(m_contour) < 4 and cv2.contourArea(m_contour) < 16):
                # print("*" * 10 + "多线程开始: ")
                starttime = time.time()
                m_rotated_box = get_min_area_bbox(
                    refined_mask_region, m_contour, self.bbox_scale_ratio
                )
                # print("7.1：", time.time() - starttime)
                starttime = time.time()
                if m_rotated_box is not None:
                    m_box_width = m_rotated_box["box_width"]
                    m_box_height = m_rotated_box["box_height"]
                    if not (
                        min(m_box_width * w, m_box_height * h) < self.shortest_length
                    ):
                        rotated_points = get_coordinates_of_rotated_box(
                            m_rotated_box, height, width
                        )
                        # print("7.2：", time.time() - starttime)

                        starttime = time.time()
                        x_min_index = np.argmin(np.sum(rotated_points, axis=-1))
                        rotated_points = np.array(
                            [
                                rotated_points[(x_min_index + 0) % 4],
                                rotated_points[(x_min_index + 1) % 4],
                                rotated_points[(x_min_index + 2) % 4],
                                rotated_points[(x_min_index + 3) % 4],
                            ]
                        )

                        m_available_mask = np.zeros_like(
                            available_region, dtype=np.uint8
                        )
                        cv2.drawContours(
                            m_available_mask,
                            [
                                m_contour,
                            ],
                            0,
                            255,
                            thickness=-1,
                        )
                        # print("7.3：", time.time() - starttime)
                        starttime = time.time()
                        # m_region_mask = cv2.bitwise_and(
                        #     available_region, available_region, mask=m_available_mask
                        # )  # 耗时比较大0.019731998443603516
                        m_region_mask = available_region
                        # print("7.4：", time.time() - starttime)
                        starttime = time.time()
                        # m_mask_count = (
                        #     m_available_mask != 0
                        # ).sum()  # 耗时比较大0.00883793830871582
                        m_mask_count = int(np.count_nonzero(m_available_mask))
                        # print("7.5：", time.time() - starttime)
                        starttime = time.time()
                        self.lock.acquire()
                        # print("7.6：", time.time() - starttime)
                        starttime = time.time()
                        tmp_socre.append(
                            float(np.sum(m_region_mask) / m_mask_count)
                        )  # 耗时比较大0.0059814453125
                        # print("7.7：", time.time() - starttime)
                        starttime = time.time()
                        tmp_points.append(rotated_points)
                        # print("7.8：", time.time() - starttime)
                        starttime = time.time()
                        self.lock.release()
                        # print("7.9：", time.time() - starttime)


def rotate_points(_points, _degree=0, _center=(0, 0)):
    """
    逆时针绕着一个点旋转点
    Args:
        _points:    需要旋转的点
        _degree:    角度
        _center:    中心点
    Returns:    旋转后的点
    """
    angle = np.deg2rad(_degree)
    rotate_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    center = np.atleast_2d(_center)
    points = np.atleast_2d(_points)
    return np.squeeze((rotate_matrix @ (points.T - center.T) + center.T).T)


def get_coordinates_of_rotated_box(_rotated_box, _height, _width):
    """
    获取旋转的矩形的对应的四个顶点坐标
    Args:
        _image:     对应的图像
        _rotated_box:   旋转的矩形
    Returns:    四个对应在图像中的坐标点
    """
    center_x = _rotated_box["center_x"]
    center_y = _rotated_box["center_y"]
    half_box_width = _rotated_box["box_width"] / 2
    half_box_height = _rotated_box["box_height"] / 2
    raw_points = np.array(
        [
            [center_x - half_box_width, center_y - half_box_height],
            [center_x + half_box_width, center_y - half_box_height],
            [center_x + half_box_width, center_y + half_box_height],
            [center_x - half_box_width, center_y + half_box_height],
        ]
    ) * (_width, _height)
    rotated_points = rotate_points(
        raw_points, _rotated_box["degree"], (center_x * _width, center_y * _height)
    )
    rotated_points[:, 0] = np.clip(rotated_points[:, 0], a_min=0, a_max=_width)
    rotated_points[:, 1] = np.clip(rotated_points[:, 1], a_min=0, a_max=_height)
    return rotated_points.astype(np.int32)


def get_min_area_bbox(_image, _contour, _scale_ratio=1.0):
    """
    获取一个contour对应的最小面积矩形
    note:主要是解决了旋转角度不合适的问题
    Args:
        _image:     bbox所在图像
        _contour:   轮廓
        _scale_ratio:      缩放比例
    Returns:    最小面积矩形的相关信息
    """
    h, w = _image.shape[:2]
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
        rotated_box = cv2.minAreaRect(scaled_contour)
    except Exception:
        return None
    if -90 <= rotated_box[2] <= -45:
        to_rotate_degree = rotated_box[2] + 90
        bbox_height, bbox_width = rotated_box[1]
    else:
        to_rotate_degree = rotated_box[2]
        bbox_width, bbox_height = rotated_box[1]
    # 几何信息归一化可以方便进行在缩放前的图像上进行操作
    to_return_rotated_box = {
        "degree": to_rotate_degree,
        "center_x": rotated_box[0][0] / w,
        "center_y": rotated_box[0][1] / h,
        "box_height": bbox_height / h,
        "box_width": bbox_width / w,
    }
    return to_return_rotated_box
