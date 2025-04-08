import time
from typing import List, Tuple

import numpy as np
import cv2
import math
from PIL import ImageDraw, ImageFont

from WORKFLOW.OTHER.surya.surya.postprocessing.fonts import get_font_path
from WORKFLOW.OTHER.surya.surya.postprocessing.util import rescale_bbox
from WORKFLOW.OTHER.surya.surya.schema import PolygonBox
from WORKFLOW.OTHER.surya.surya.settings import settings
from WORKFLOW.OTHER.surya.surya.postprocessing.text import get_text_size


def keep_largest_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        box = box_obj.bbox
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            other_box_area = (other_box[2] - other_box[0]) * (
                other_box[3] - other_box[1]
            )
            if box == other_box:
                continue
            # find overlap percentage
            overlap = box_obj.intersection_pct(other_box_obj)
            if overlap > 0.9 and box_area < other_box_area:
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def clean_contained_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        box = box_obj.bbox
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            if box == other_box:
                continue
            if (
                box[0] >= other_box[0]
                and box[1] >= other_box[1]
                and box[2] <= other_box[2]
                and box[3] <= other_box[3]
            ):
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def get_dynamic_thresholds(linemap, text_threshold, low_text, typical_top10_avg=0.7):
    # Find average intensity of top 10% pixels
    # Do top 10% to account for pdfs that are mostly whitespace, etc.
    flat_map = linemap.flatten()
    sorted_map = np.sort(flat_map)[::-1]
    top_10_count = int(np.ceil(len(flat_map) * 0.1))
    top_10 = sorted_map[:top_10_count]
    avg_intensity = np.mean(top_10)

    # Adjust thresholds based on normalized intensityy
    scaling_factor = min(1, avg_intensity / typical_top10_avg) ** (1 / 2)

    low_text = max(low_text * scaling_factor, 0.1)
    text_threshold = max(text_threshold * scaling_factor, 0.15)

    low_text = min(low_text, 0.6)
    text_threshold = min(text_threshold, 0.8)
    return text_threshold, low_text


def detect_boxes(linemap, text_threshold, low_text):
    # From CRAFT - https://github.com/clovaai/CRAFT-pytorch
    # prepare data
    factor = 0.5
    linemap = cv2.resize(
        linemap, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
    )
    img_h, img_w = linemap.shape

    # ss = time.time()
    text_threshold, low_text = get_dynamic_thresholds(linemap, text_threshold, low_text)
    # print("3.1: " + "*" * 5, round(time.time() - ss, 5))

    # ss = time.time()
    ret, text_score = cv2.threshold(linemap, low_text, 1, cv2.THRESH_BINARY)
    # print("3.2: " + "*" * 5, round(time.time() - ss, 5))

    # ss = time.time()
    text_score_comb = np.clip(text_score, 0, 1).astype(np.uint8)
    # print("3.3: " + "*" * 5, round(time.time() - ss, 5))

    # ss = time.time()
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb, connectivity=4
    )
    # print("3.4: " + "*" * 5, round(time.time() - ss, 5))

    # ss = time.time()
    det = []
    confidences = []
    max_confidence = 0
    mask = np.zeros_like(linemap, dtype=np.uint8)

    for k in range(1, label_count):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(linemap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(linemap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # boundary checks
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = (
            np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            .transpose()
            .reshape(-1, 2)
        )
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        mask.fill(0)
        cv2.fillPoly(mask, [np.int32(box)], 1)

        roi = np.where(mask == 1, linemap, 0)
        confidence = np.mean(roi[roi != 0])

        if confidence > max_confidence:
            max_confidence = confidence

        confidences.append(confidence)
        det.append(box)

    if max_confidence > 0:
        confidences = [c / max_confidence for c in confidences]
    # print("3.5: " + "*" * 5, round(time.time() - ss, 5))

    det = (np.array(det) / factor).tolist()
    labels = cv2.resize(
        labels.astype(np.float32),
        tuple(np.array(np.array(text_score_comb.shape[::-1]) / factor, dtype=int)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
    return det, labels, confidences


def get_detected_boxes(textmap, text_threshold=None, low_text=None) -> List[PolygonBox]:
    if text_threshold is None:
        text_threshold = settings.DETECTOR_TEXT_THRESHOLD

    if low_text is None:
        low_text = settings.DETECTOR_BLANK_THRESHOLD

    textmap = textmap.copy()
    textmap = textmap.astype(np.float32)
    boxes, labels, confidences = detect_boxes(textmap, text_threshold, low_text)
    # From point form to box form
    boxes = [
        PolygonBox(polygon=box, confidence=confidence)
        for box, confidence in zip(boxes, confidences)
    ]
    return boxes


def get_and_clean_boxes(
    textmap, processor_size, image_size, text_threshold=None, low_text=None
) -> List[PolygonBox]:
    bboxes = get_detected_boxes(textmap, text_threshold, low_text)
    for bbox in bboxes:
        bbox.rescale(processor_size, image_size)
        bbox.fit_to_bounds([0, 0, image_size[0], image_size[1]])

    bboxes = clean_contained_boxes(bboxes)
    return bboxes


def draw_bboxes_on_image(bboxes, image, labels=None):
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=1)

    return image


def draw_polys_on_image(
    corners, image, labels=None, box_padding=-1, label_offset=1, label_font_size=10
):
    draw = ImageDraw.Draw(image)
    font_path = get_font_path()
    label_font = ImageFont.truetype(font_path, label_font_size)

    for i in range(len(corners)):
        poly = corners[i]
        poly = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(poly, outline="red", width=1)

        if labels is not None:
            label = labels[i]
            text_position = (
                min([p[0] for p in poly]) + label_offset,
                min([p[1] for p in poly]) + label_offset,
            )
            text_size = get_text_size(label, label_font)
            box_position = (
                text_position[0] - box_padding + label_offset,
                text_position[1] - box_padding + label_offset,
                text_position[0] + text_size[0] + box_padding + label_offset,
                text_position[1] + text_size[1] + box_padding + label_offset,
            )
            draw.rectangle(box_position, fill="white")
            draw.text(text_position, label, fill="red", font=label_font)

    return image
