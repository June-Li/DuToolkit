import os
import sys
import cv2
import copy


cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/YOLO/YOLOv3/")))

from WORKFLOW.DET.TableCellDet.v0.detector_table_cell import Detector
from WORKFLOW.DET.TableCellDet.v0.table_cell_postprocess import merge_line_cell

path = cur_dir + "/test_data/my_imgs_0/1.jpg"
im0s = cv2.imread(path)
detector = Detector(
    os.path.abspath(root_dir + "/MODEL/DET/YOLO/YOLOv3/TableCellDet/20210531/best.pt")
)
boxes, _, _ = detector.inference(im0s)

merge_cls = merge_line_cell()
_, _, merge_boxes = merge_cls.merge_line(
    im0s.copy(),
    copy.deepcopy(boxes),
    iou_threshold=-0.1,
    distance_threshold="calculate",
    show_flag=False,
)  # 0.025

fill_boxes_ = merge_cls.fill_table(copy.deepcopy(merge_boxes))
_, _, merge_boxes = merge_cls.merge_line(
    im0s.copy(),
    copy.deepcopy(fill_boxes_),
    iou_threshold=-0.1,
    distance_threshold="calculate",
    show_flag=False,
)

for box in merge_boxes:
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(im0s, c1, c2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

out_path = "/".join(path.replace("/test_data/", "/test_out/").split("/")[:-1])
if not os.path.exists(out_path):
    os.makedirs(out_path)
cv2.imwrite(path.replace("/test_data/", "/test_out/"), im0s)
print(merge_boxes)
