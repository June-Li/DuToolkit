import cv2
import os
import numpy as np
import time
import copy
import xlsxwriter
from docx import Document


class merge_line_cell:
    def __init__(self):
        pass

    # 公用的后处理函数
    def cal_iou(self, box_1, box_2):
        iou, overlap_area, flag = 0, 0, False
        min_x = min(box_1[0], box_1[2], box_2[0], box_2[2])
        max_x = max(box_1[0], box_1[2], box_2[0], box_2[2])
        min_y = min(box_1[1], box_1[3], box_2[1], box_2[3])
        max_y = max(box_1[1], box_1[3], box_2[1], box_2[3])
        box_1_w = abs(box_1[0] - box_1[2])
        box_1_h = abs(box_1[1] - box_1[3])
        box_2_w = abs(box_2[0] - box_2[2])
        box_2_h = abs(box_2[1] - box_2[3])
        merge_w = max_x - min_x
        merge_h = max_y - min_y
        overlap_w = box_1_w + box_2_w - merge_w
        overlap_h = box_1_h + box_2_h - merge_h
        if overlap_h > 0 and overlap_w > 0:
            box_1_area = box_1_w * box_1_h
            box_2_area = box_2_w * box_2_h
            overlap_area = overlap_w * overlap_h
            iou = overlap_area / (box_1_area + box_2_area - overlap_area)
            if overlap_w > 10 or overlap_h > 10:
                flag = True
        return iou, flag

    def txt2cell(self, merge_boxes, text_recs_cell, rec_res):
        text_list = []
        merge_index_set = []
        for i in range(len(merge_boxes)):
            text_list.append("")

        for index_c, i in enumerate(text_recs_cell):
            table_cells_boxes = [int(i[0][0]), int(i[0][1]), int(i[2][0]), int(i[2][1])]
            max_index = -99999
            max_iou = -99999
            temp_set = []
            for index_m, box in enumerate(merge_boxes):
                if (
                    box[3] < table_cells_boxes[1]
                    or box[1] > table_cells_boxes[3]
                    or box[2] < table_cells_boxes[0]
                    or box[0] > table_cells_boxes[2]
                ):
                    continue
                if (
                    box[0] <= table_cells_boxes[0]
                    and box[1] <= table_cells_boxes[1]
                    and box[2] >= table_cells_boxes[2]
                    and box[3] >= table_cells_boxes[3]
                ):
                    max_iou = 1
                    max_index = index_m
                    break
                iou, _ = self.cal_iou(box, table_cells_boxes)
                if iou > 0.1:
                    temp_set.append(index_m)
                if iou > max_iou:
                    max_iou = iou
                    max_index = index_m
            merge_index_set.append(temp_set)
            if max_iou > 0.1:
                text_list[max_index] += rec_res[index_c][0]

        no_merge_index_set = [i for i in range(len(merge_boxes))]
        for i in merge_index_set:
            if len(i) >= 2:
                for index in i:
                    if index in no_merge_index_set:
                        no_merge_index_set.remove(index)

        merge_boxes_result = []
        text_list_result = []

        for index in no_merge_index_set:
            merge_boxes_result.append(merge_boxes[index])
            text_list_result.append(text_list[index])

        for i in merge_index_set:
            if len(i) < 2:
                continue
            temp_boxes = []
            temp_text = ""
            for index in i:
                temp_boxes.append(merge_boxes[index])
                temp_text += text_list[index]
            temp_boxes = np.array(temp_boxes)
            x_0 = np.min(temp_boxes[:, [0, 2]])
            y_0 = np.min(temp_boxes[:, [1, 3]])
            x_1 = np.max(temp_boxes[:, [0, 2]])
            y_1 = np.max(temp_boxes[:, [1, 3]])
            merge_boxes_result.append([x_0, y_0, x_1, y_1])
            text_list_result.append(temp_text)
        return merge_boxes_result, text_list_result

    def iou_matrix(self, boxes_list_1, boxes_list_2):
        iou_matrix_ = np.zeros((len(boxes_list_1), len(boxes_list_2)))
        for index_1, box_1 in enumerate(boxes_list_1):
            for index_2, box_2 in enumerate(boxes_list_2):
                iou, _ = self.cal_iou(box_1, box_2)
                if iou > 0:
                    iou_matrix_[index_1, index_2] = iou
        return iou_matrix_

    # yolov3的结果用的后处理函数
    def find_connected(self, isConnected):
        """
        深度优先搜索算法，把相近的线划分为同一个连通域
        """

        def dfs(i: int, province_list):
            for j in range(provinces):
                if isConnected[i][j] == 1 and j not in visited:
                    visited.add(j)
                    province_list.append(j)
                    dfs(j, province_list)

        provinces = len(isConnected)
        visited = set()
        circles = 0
        provinces_list = []
        for i in range(provinces):
            if i not in visited:
                province_list = []
                dfs(i, province_list)
                circles += 1
                provinces_list.append(province_list)

        return circles, provinces_list

    def calculate_distance_threshold(self, boxes):
        dial_list = []
        height_list = []
        for i in range(100):
            dial_list.append(0)
            height_list.append([])
        for box in boxes:
            try:
                dial_list[int((abs(box[3] - box[1])) / 50)] += 1
                height_list[int((abs(box[3] - box[1])) / 50)].append(
                    abs(box[3] - box[1])
                )
            except:
                dial_list[-1] += 1
                height_list[-1].append(abs(box[3] - box[1]))
        index = int(np.argmax(dial_list))
        distance_threshold = np.average(height_list[index]) / 3

        if distance_threshold < 20:
            return distance_threshold
        else:
            return 20

    def merge_h_lines(self, h_line, iou_threshold, distance_threshold, boxes):
        """
        深度优先搜索，所以先构建邻接矩阵
        """
        isConnected = np.zeros((len(h_line), len(h_line)))
        for index_one in range(len(h_line)):
            for index_two in range(len(h_line)):
                if index_two < index_one:
                    continue
                elif index_two == index_one:
                    isConnected[index_one][index_two] = 1
                    continue
                line_1 = h_line[index_one]
                line_2 = h_line[index_two]
                union = max(
                    line_1[0][0], line_1[1][0], line_2[0][0], line_2[1][0]
                ) - min(line_1[0][0], line_1[1][0], line_2[0][0], line_2[1][0])
                total = abs(line_1[0][0] - line_1[1][0]) + abs(
                    line_2[0][0] - line_2[1][0]
                )
                intersect = total - union
                iou = intersect / union
                distance = abs(line_1[0][1] - line_2[0][1])  # / np.shape(image)[0]
                if iou > iou_threshold and distance < distance_threshold:
                    isConnected[index_one][index_two] = 1
                    isConnected[index_two][index_one] = 1
        circles, provinces_list = self.find_connected(isConnected)

        merge_h_line_list = []
        for province_list in provinces_list:
            x_set = []
            y_set = []
            for index in province_list:
                line = h_line[index]
                x_set.append(line[0][0])
                x_set.append(line[1][0])
                y_set.append(line[0][1])
            merge_h_line_list.append(
                [[min(x_set), np.average(y_set)], [max(x_set), np.average(y_set)]]
            )
            for index in province_list:
                if h_line[index][2][0] == "up":
                    boxes[h_line[index][3]][1] = int(np.average(y_set))
                elif h_line[index][2][0] == "down":
                    boxes[h_line[index][3]][3] = int(np.average(y_set))
                else:
                    raise ValueError

        merge_h_line_list = np.array(merge_h_line_list, dtype=int)

        for index_, box_ in enumerate(boxes):
            if box_[0] > box_[2]:
                boxes[index_][0] = box_[2]
                boxes[index_][2] = box_[0]
            if box_[1] > box_[3]:
                boxes[index_][1] = box_[3]
                boxes[index_][3] = box_[1]

        return merge_h_line_list, boxes

    def merge_v_lines(self, v_line, iou_threshold, distance_threshold, boxes):
        """
        深度优先搜索，所以先构建邻接矩阵
        """
        isConnected = np.zeros((len(v_line), len(v_line)))
        for index_one in range(len(v_line)):
            for index_two in range(len(v_line)):
                if index_two < index_one:
                    continue
                elif index_two == index_one:
                    isConnected[index_one][index_two] = 1
                    continue
                line_1 = v_line[index_one]
                line_2 = v_line[index_two]
                union = max(
                    line_1[0][1], line_1[1][1], line_2[0][1], line_2[1][1]
                ) - min(line_1[0][1], line_1[1][1], line_2[0][1], line_2[1][1])
                total = abs(line_1[0][1] - line_1[1][1]) + abs(
                    line_2[0][1] - line_2[1][1]
                )
                intersect = total - union
                iou = intersect / union
                distance = abs(line_1[0][0] - line_2[0][0])  # / np.shape(image)[0]
                if iou > iou_threshold and distance < distance_threshold:
                    isConnected[index_one][index_two] = 1
                    isConnected[index_two][index_one] = 1

        circles, provinces_list = self.find_connected(isConnected)

        merge_v_line_list = []
        for province_list in provinces_list:
            x_set = []
            y_set = []
            for index in province_list:
                line = v_line[index]
                x_set.append(line[0][0])
                y_set.append(line[0][1])
                y_set.append(line[1][1])
            merge_v_line_list.append(
                [[np.average(x_set), min(y_set)], [np.average(x_set), max(y_set)]]
            )
            for index in province_list:
                if v_line[index][2][0] == "left":
                    boxes[v_line[index][3]][0] = int(np.average(x_set))
                elif v_line[index][2][0] == "right":
                    boxes[v_line[index][3]][2] = int(np.average(x_set))
                else:
                    raise ValueError

        merge_v_line_list = np.array(merge_v_line_list, dtype=int)
        for index_, box_ in enumerate(boxes):
            if box_[0] > box_[2]:
                boxes[index_][0] = box_[2]
                boxes[index_][2] = box_[0]
            if box_[1] > box_[3]:
                boxes[index_][1] = box_[3]
                boxes[index_][3] = box_[1]

        return merge_v_line_list, boxes

    def merge_line(
        self,
        image,
        boxes,
        iou_threshold=-0.1,
        distance_threshold="calculate",
        show_flag=False,
    ):
        """
        image:
            这个是经过检测从原图抠出来的patch
        boxes：
            格式为[[x0, y0, x1, y1],……]，即左上右下，是相对于抠出来patch的坐标位置
        iou_threshold：
            iou即把横线移到同一水平后的交并比（把竖线移到竖方向同一位置后的交并比），
            两条线iou小于阈值可合并（需同时满足distance_threshold）。
            eg：
                calculate：表示需要通过cell的height基础出阈值。
                20：表示直接给出阈值，不需要计算。
        distance_threshold：
            横线是两条线y的距离相对图像h的比例阈值（竖线是两条线x的距离相对图像h的比例阈值，比例要和横线保持一致，所以也用h），
            两条小距离小于阈值可合并（需同时满足iou_threshold）
        show_flag：
            是否显示图片演示
        """
        # boxes = self.fill_table(image, boxes)
        start_time = time.time()
        h_line = []
        v_line = []
        box_image = image.copy()
        new_box_image = image.copy()
        index = 0
        for box in boxes:
            if abs(box[0] - box[2]) > 0:
                h_line.append([[box[0], box[1]], [box[2], box[1]], ["up"], index])
            if abs(box[0] - box[2]) > 0:
                h_line.append([[box[0], box[3]], [box[2], box[3]], ["down"], index])
            if abs(box[1] - box[3]) > 0:
                v_line.append([[box[0], box[1]], [box[0], box[3]], ["left"], index])
            if abs(box[1] - box[3]) > 0:
                v_line.append([[box[2], box[1]], [box[2], box[3]], ["right"], index])
            if show_flag:
                cv2.rectangle(
                    box_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
                )
            index += 1
        if distance_threshold == "calculate":
            distance_threshold = self.calculate_distance_threshold(boxes)
        else:
            distance_threshold = np.float(distance_threshold)

        merge_h_line_list, boxes = self.merge_h_lines(
            h_line, iou_threshold, distance_threshold, boxes
        )
        merge_v_line_list, boxes = self.merge_v_lines(
            v_line, iou_threshold, distance_threshold, boxes
        )

        for box in copy.deepcopy(boxes):
            if (
                abs(box[0] - box[2]) < 3
                or abs(box[1] - box[3]) < 3
                or abs(box[0] - box[2]) * abs(box[1] - box[3]) < 9
            ):
                boxes.remove(box)

        if show_flag:
            for line in merge_h_line_list:
                cv2.line(image, tuple(line[0]), tuple(line[1]), (0, 0, 255), 2)
            for line in merge_v_line_list:
                cv2.line(image, tuple(line[0]), tuple(line[1]), (0, 0, 255), 2)
            for box in boxes:
                cv2.rectangle(
                    new_box_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("box_image", box_image)
            cv2.imshow("image", image)
            cv2.imshow("new_box_image", new_box_image)
            cv2.waitKey()
        print("merge use time: ", time.time() - start_time)
        return merge_h_line_list, merge_v_line_list, boxes

    def remove_duplicate(self, boxes, table_h, table_w):
        start_time = time.time()
        show_box = np.ones((table_h, table_w, 3), dtype=np.uint8) * 255
        remove_dup_boxes = []
        for index_one in range(len(boxes)):
            for index_two in range(len(boxes)):
                if index_two > index_one:
                    iou, adjust_flag = self.cal_iou(
                        boxes[index_one].copy(), boxes[index_two].copy()
                    )
                    if adjust_flag:
                        min_x = min(
                            boxes[index_one][0],
                            boxes[index_one][2],
                            boxes[index_two][0],
                            boxes[index_two][2],
                        )
                        max_x = max(
                            boxes[index_one][0],
                            boxes[index_one][2],
                            boxes[index_two][0],
                            boxes[index_two][2],
                        )
                        min_y = min(
                            boxes[index_one][1],
                            boxes[index_one][3],
                            boxes[index_two][1],
                            boxes[index_two][3],
                        )
                        max_y = max(
                            boxes[index_one][1],
                            boxes[index_one][3],
                            boxes[index_two][1],
                            boxes[index_two][3],
                        )
                        h, w = max_y - min_y, max_x - min_x
                        mask_img = np.zeros((h, w))
                        mask_img[
                            boxes[index_one][1] - min_y : boxes[index_one][3] - min_y,
                            boxes[index_one][0] - min_x : boxes[index_one][2] - min_x,
                        ] += 1
                        mask_img[
                            boxes[index_two][1] - min_y : boxes[index_two][3] - min_y,
                            boxes[index_two][0] - min_x : boxes[index_two][2] - min_x,
                        ] += 1
                        mask_img[mask_img < 2] = 0
                        overlap_area_index = np.argwhere(mask_img == 2)
                        overlap_area_min_x = min(overlap_area_index[:, 1])
                        overlap_area_max_x = max(overlap_area_index[:, 1])
                        overlap_area_min_y = min(overlap_area_index[:, 0])
                        overlap_area_max_y = max(overlap_area_index[:, 0])

                        point_x_list = [
                            boxes[index_one][0] - min_x,
                            boxes[index_one][2] - min_x,
                            overlap_area_min_x,
                            overlap_area_max_x,
                        ]
                        point_y_list = [
                            boxes[index_one][1] - min_y,
                            boxes[index_one][3] - min_y,
                            overlap_area_min_y,
                            overlap_area_max_y,
                        ]
                        point_list = []
                        for one in point_x_list:
                            for two in point_y_list:
                                point_list.append([one, two])
                        point_list = np.unique(point_list, axis=0)

                        max_box = []
                        max_box_area = -1
                        for one in range(len(point_list)):
                            for two in range(len(point_list)):
                                if two <= one:
                                    continue
                                x_0 = min(point_list[one][0], point_list[two][0]) + 1
                                y_0 = min(point_list[one][1], point_list[two][1]) + 1
                                x_1 = max(point_list[one][0], point_list[two][0]) - 1
                                y_1 = max(point_list[one][1], point_list[two][1]) - 1
                                if np.sum(mask_img[y_0:y_1, x_0:x_1]) < 10:
                                    box_area = abs(x_1 - x_0) * abs(y_1 - y_0)
                                    if max_box_area < box_area:
                                        max_box_area = box_area
                                        max_box = [
                                            min(x_0 + min_x, x_1 + min_x),
                                            min(y_0 + min_y, y_1 + min_y),
                                            max(x_0 + min_x, x_1 + min_x),
                                            max(y_0 + min_y, y_1 + min_y),
                                        ]
                        boxes[index_one] = max_box
            remove_dup_boxes.append(boxes[index_one])
        print("remove dup boxes use time: ", time.time() - start_time)
        # for box in remove_dup_boxes:
        #     cv2.rectangle(show_box, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # cv2.imshow('image', show_box)
        # cv2.waitKey()
        return remove_dup_boxes

    def fill_table(self, boxes):
        x_set, y_set = [], []
        for box in boxes:
            x_set.append(box[0])
            x_set.append(box[2])
            y_set.append(box[1])
            y_set.append(box[3])

        table_min_x = min(x_set)
        table_max_x = max(x_set)
        table_min_y = min(y_set)
        table_max_y = max(y_set)
        image_binary = (
            np.ones(
                (table_max_y - table_min_y, table_max_x - table_min_x), dtype=np.uint8
            )
            * 255
        )
        offset_boxes = []
        for box in boxes:
            offset_boxes.append(
                [
                    box[0] - table_min_x,
                    box[1] - table_min_y,
                    box[2] - table_min_x,
                    box[3] - table_min_y,
                ]
            )
        remove_dup_boxes = self.remove_duplicate(
            offset_boxes, table_max_y - table_min_y, table_max_x - table_min_x
        )
        boxes = []
        for box in remove_dup_boxes:
            boxes.append(
                [
                    box[0] + table_min_x,
                    box[1] + table_min_y,
                    box[2] + table_min_x,
                    box[3] + table_min_y,
                ]
            )
        for box in remove_dup_boxes:
            image_binary[box[1] : box[3], box[0] : box[2]] = 0

        start_time = time.time()
        new_box = []
        h, w = np.shape(image_binary)[0], np.shape(image_binary)[1]

        i, j = 0, 0
        row_block = 1
        for _ in range(h):
            if not np.sum(image_binary[j : min(j + row_block, h), :]):
                i = 0
                j = min(j + row_block, h)
                continue
            for _ in range(w):
                if image_binary[j, i]:
                    step_x = 0
                    step_y = 0
                    for ii in range(i, w):
                        if image_binary[j, ii]:
                            step_x += 1
                        else:
                            break
                    for jj in range(j, h):
                        if np.average(image_binary[jj, i : i + step_x]) == 255:
                            step_y += 1
                        else:
                            break
                    # print(i, j, step_x, step_y)
                    image_binary[j : j + step_y, i : i + step_x] = 0
                    new_box.append([i, j, i + step_x, j + step_y])
                i += 1
                if i >= w:
                    i = 0
                    break
            j += 1
            if j >= h:
                break

        print("use time fill: ", time.time() - start_time)

        new_box_reoffset = []
        for box in new_box:
            new_box_reoffset.append(
                [
                    box[0] + table_min_x,
                    box[1] + table_min_y,
                    box[2] + table_min_x,
                    box[3] + table_min_y,
                ]
            )

        boxes = boxes + new_box_reoffset
        return boxes

    def table_structure(self, out_path, boxes, text_list):
        start_time = time.time()
        # 按照x0，y0，x1，y1的顺序进行排序，如果要给box排序的话取消注释，但是没有把握尽量不要排序，因为可能影响后边的text和box的对应关系
        text_list = np.array(text_list)
        boxes = np.array([[box[1], box[0], box[3], box[2]] for box in boxes])
        temp_boxes = np.transpose(boxes)
        temp_boxes = np.flipud(temp_boxes)
        sort_index = np.lexsort(temp_boxes)
        boxes = boxes[sort_index]
        text_list = text_list[sort_index]
        boxes = [[box[1], box[0], box[3], box[2]] for box in boxes]
        boxes, text_list = list(boxes), list(text_list)

        x_set, y_set = [], []
        for box in boxes:
            x_set.append(box[0])
            x_set.append(box[2])
            y_set.append(box[1])
            y_set.append(box[3])
        table_min_x, table_max_x, table_min_y, table_max_y = (
            min(x_set),
            max(x_set),
            min(y_set),
            max(y_set),
        )
        offset_boxes = []
        w_list, h_list = [], []
        for box in boxes:
            offset_boxes.append(
                [
                    box[0] - table_min_x,
                    box[1] - table_min_y,
                    box[2] - table_min_x,
                    box[3] - table_min_y,
                ]
            )
            w_list.append(abs((box[0] - table_min_x) - (box[2] - table_min_x)))
            h_list.append(abs((box[1] - table_min_y) - (box[3] - table_min_y)))
        scale = min(min(w_list), min(h_list))
        h, w = table_max_y - table_min_y + 1, table_max_x - table_min_x + 1

        # '''写入excel和word的代码，需要的话取消注释
        f = xlsxwriter.Workbook(out_path.replace(".html", ".xlsx"))  # 创建excel文件
        worksheet1 = f.add_worksheet("sheet1")
        # color_list = ['#FFB6C1', '#FFC0CB', '#DC143C','#FFF0F5','#DB7093','#FF69B4','#FF1493','#C71585' ,'#DA70D6','#D8BFD8','#DDA0DD','#EE82EE','#FF00FF','#FF00FF','#8B008B','#800080','#BA55D3','#9400D3']
        color_list = ["#FFFFFF"]

        None_list = [None for i in range(w // scale)]
        for i in range(h // scale):
            worksheet1.write_row(
                i, 0, None_list, f.add_format({"fg_color": "#FFFFFF", "border": 2})
            )
        # worksheet1.set_column(0, w*10, 1)
        [worksheet1.set_column(i, 1) for i in range(w // scale)]
        [worksheet1.set_row(i, 20) for i in range(h // scale)]
        # box_file = open('/Volumes/my_disk/company/sensedeal/buffer_disk/buffer_8/1.txt', 'a+')
        index = 0
        for box in offset_boxes:
            worksheet1.write(
                box[1] // scale,
                box[0] // scale,
                text_list[index],
                f.add_format(
                    {
                        "fg_color": color_list[np.random.randint(0, len(color_list))],
                        "border": 2,
                    }
                ),
            )
            worksheet1.merge_range(
                box[1] // scale,
                box[0] // scale,
                box[3] // scale - 1,
                box[2] // scale - 1,
                text_list[index],
                f.add_format(
                    {
                        "fg_color": color_list[np.random.randint(0, len(color_list))],
                        "border": 2,
                    }
                ),
            )
            index += 1
            # box_file.write(' '.join([str(i) for i in box]) + '\n')
        f.close()

        # 写入word开始
        # doc = Document()
        # table = doc.add_table(h // scale, w // scale, style='Table Grid')
        #
        # index = 0
        # for box in offset_boxes:
        #     table.cell(box[1] // scale, box[0] // scale).merge(table.cell(box[3] // scale - 1, box[2] // scale - 1)).text = text_list[index]
        #     index += 1
        # doc.save(out_path.replace('.html', '.docx'))
        # '''

        # '''
        # 写入html开始
        html_table_list = []
        for i in range(1 if h // scale == 0 else h // scale):
            html_h_list = []
            for j in range(1 if w // scale == 0 else w // scale):
                html_h_list.append("")
            html_table_list.append(html_h_list)

        index = 0
        for box in offset_boxes:
            x_0, y_0, x_1, y_1 = (
                box[0] // scale,
                box[1] // scale,
                box[2] // scale,
                box[3] // scale,
            )
            cell_str = "<td "
            cell_str = cell_str + "class=" + '"' + "tg-0lax" + '" '
            cell_str = (
                cell_str + "rowspan=" + '"' + str(y_1 - y_0) + '" '
            )  # 向下融合cell的数量
            cell_str = (
                cell_str + "colspan=" + '"' + str(x_1 - x_0) + '" '
            )  # 向右融合cell的数量
            cell_str = (
                cell_str + "height=" + '"' + str(box[3] - box[1]) + '" '
            )  # 设置cell的宽
            cell_str = (
                cell_str + "width=" + '"' + str(box[2] - box[0]) + '" '
            )  # 设置cell的高
            cell_str = cell_str + ">"
            cell_str = cell_str + text_list[index]  # 文本内容
            cell_str = cell_str + "</td>"  # 结束符
            html_table_list[y_0][x_0] = cell_str
            index += 1
        html_file = open(out_path, "w")
        html_file.write(self.html_configuration()[0])
        for i in html_table_list:
            html_file.write("<tr>\n")
            for j in i:
                if j != "":
                    html_file.write(j + "\n")
            html_file.write("</tr>\n")
        html_file.write(self.html_configuration()[1])
        html_file.close()
        print("structure use time: ", time.time() - start_time)
        # '''

        # txt cell坐标点
        x_point_dict, y_point_dict = self.cell_num_type(offset_boxes, scale)
        txt_cell_coordinate_list = {}
        for index, box in enumerate(offset_boxes):
            x_0, y_0, x_1, y_1 = (
                x_point_dict[box[0] // scale],
                y_point_dict[box[1] // scale],
                x_point_dict[box[2] // scale],
                y_point_dict[box[3] // scale],
            )
            txt_cell_coordinate_list[index] = [[y_0, x_0, y_1, x_1]]
            txt_cell_coordinate_list[index].append(text_list[index])
        return txt_cell_coordinate_list

    def cell_num_type(self, offset_boxes, scale):
        """
        输入的是融合的box占了多少最小单元，但是实际标记的时候只需要告诉客户box的个数，而不需要知道融合了多少个最小单元。
        所以这个函数进一步处理，得到计数方式的cell的左上角右下角
        :return: 计数方式的cell的左上角右下角
        """
        x_point_list = []
        y_point_list = []
        for index, box in enumerate(offset_boxes):
            x_point_list.append(box[0] // scale)
            x_point_list.append(box[2] // scale)
            y_point_list.append(box[1] // scale)
            y_point_list.append(box[3] // scale)
        x_point_list, y_point_list = sorted(list(set(x_point_list))), sorted(
            list(set(y_point_list))
        )
        x_point_dict = {j: i for i, j in enumerate(x_point_list)}
        y_point_dict = {j: i for i, j in enumerate(y_point_list)}
        return x_point_dict, y_point_dict

    def html_configuration(self):
        html_head = (
            "<!DOCTYPE html>\n"
            + '<html lang="en">\n'
            + "<head>\n"
            + '    <meta charset="UTF-8">\n'
            + "    <title>Title</title>\n"
            + "</head>\n"
            + "<body>\n"
            + '<style type="text/css">\n'
            + ".tg  {border-collapse:collapse;border-spacing:0;}\n"
            + ".tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;\n"
            + "  overflow:hidden;padding:10px 5px;word-break:normal;}\n"
            + ".tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;\n"
            + "  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}\n"
            + ".tg .tg-0lax{text-align:left;vertical-align:top}\n"
            + "</style>\n"
            + '<table class="tg">\n'
            + "    <tbody>\n"
        )
        html_tail = "    </tbody>\n" + "</table>\n" + "</body>\n" + "</html>\n"
        return [html_head, html_tail]

    # 文字检测DB的结果用的后处理函数
    def get_gap(self, ori_img, boxes, radio_255_0=0.8, radio_gap=0.0067):
        # start_time = time.time()
        h, w, _ = np.shape(ori_img)
        img = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            if box[2] - box[0] > 25:
                x_0, x_1 = int(box[0] + 10), int(box[2] - 10)
            else:
                continue
            if box[3] - box[1] > 20:
                y_0, y_1 = int(box[1] + 5), int(box[3] - 5)
            else:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 4), int(
                    (box[1] + box[3]) // 2 + 4
                )
            box = [x_0, y_0, x_1, y_1]
            img[box[1] : box[3], box[0] : box[2]] = 255

        h_lines = []
        for i in range(h):
            count_255 = np.sum(np.array([img[i, :] == 255], dtype=int))
            count_0 = w - np.sum(np.array([img[i, :] == 255], dtype=int))
            radio_0 = count_0 / (count_255 + count_0)
            if radio_0 > radio_255_0:
                h_lines.append(i)
        v_lines = []
        for i in range(w):
            count_255 = np.sum(
                np.array([img[int(h * 0) : int(h * 0.9), i] == 255], dtype=int)
            )
            count_0 = w - np.sum(
                np.array([img[int(h * 0) : int(h * 0.9), i] == 255], dtype=int)
            )
            radio_0 = count_0 / (count_255 + count_0)
            if radio_0 > radio_255_0:
                v_lines.append(i)

        # debug start
        show_img = ori_img.copy()
        for line in v_lines:
            cv2.line(show_img, (line, 0), (line, h), (0, 0, 255), 2)
        cv2.imwrite("./buffer/jj.jpg", show_img)
        cv2.imwrite("./buffer/tt.jpg", ori_img)
        # debug end

        h_set = []
        pre = -999
        for i in h_lines:
            if abs(i - pre) > max(h * radio_gap, 3):
                h_set.append([i])
                pre = i
            else:
                if len(h_set[-1]) >= h * radio_gap * 25:
                    h_set.append([])
                h_set[-1].append(i)
                pre = i
        v_set = []
        pre = -999
        for i in v_lines:
            if abs(i - pre) > max(w * radio_gap, 3):
                v_set.append([i])
                pre = i
            else:
                if len(v_set[-1]) >= w * radio_gap * 25:
                    v_set.append([])
                v_set[-1].append(i)
                pre = i

        h_set_compress = []
        for i in h_set:
            if len(i) > h * radio_gap:
                h_set_compress.append(int(np.average(i)))
        if len(h_set_compress) == 0:
            h_set_compress.append(1)
            h_set_compress.append(h - 2)
            h_set_compress.sort()
        else:
            if abs(min(h_set_compress) - 1) > h * radio_gap * 5:
                h_set_compress.append(1)
            if abs(max(h_set_compress) - h) > h * radio_gap * 5:
                h_set_compress.append(h - 2)
            h_set_compress.sort()

        v_set_compress = []
        for i in v_set:
            if len(i) > w * radio_gap:
                v_set_compress.append(int(np.average(i)))
        if len(v_set_compress) == 0:
            v_set_compress.append(1)
            v_set_compress.append(w - 2)
            v_set_compress.sort()
        else:
            if abs(min(v_set_compress) - 1) > w * radio_gap * 5:
                v_set_compress.append(1)
            if abs(max(v_set_compress) - w) > w * radio_gap * 5:
                v_set_compress.append(w - 2)
            v_set_compress.sort()

        # for i in h_set_compress:
        #     img[i, :] = 125
        # for i in v_set_compress:
        #     img[:, i] = 125
        # cv2.imshow('ii', img)
        # cv2.waitKey()

        # print('get_gap() use time: ', time.time()-start_time)
        return h_set_compress, v_set_compress

    def get_boxes(self, cols, rows):
        boxes = []
        for index_row in range(len(rows) - 1):
            for index_col in range(len(cols) - 1):
                boxes.append(
                    [
                        cols[index_col],
                        rows[index_row],
                        cols[index_col + 1],
                        rows[index_row + 1],
                    ]
                )
        return boxes

    def get_merge_boxes(
        self,
        img,
        boxes,
        iou_threshold=-0.1,
        distance_threshold="calculate",
        show_flag=False,
        radio_255_0=0.8,
        radio_gap=0.0067,
    ):
        h_set_compress, v_set_compress = self.get_gap(
            img, boxes, radio_255_0, radio_gap
        )
        boxes = self.get_boxes(v_set_compress, h_set_compress)
        _, _, boxes = self.merge_line(
            img, boxes, iou_threshold, distance_threshold, show_flag
        )

        return boxes

    # def line_detection(self, image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)
    #     bw = cv2.bitwise_not(bw)
    #     ## To visualize image after thresholding ##
    #     # cv2.imshow("bw",bw)
    #     # cv2.waitKey(0)
    #     ###########################################
    #     horizontal = bw.copy()
    #     vertical = bw.copy()
    #     img = image.copy()
    #     # [horizontal lines]
    #     # Create structure element for extracting horizontal lines through morphology operations
    #     horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 1))
    #
    #     # Apply morphology operations
    #     horizontal = cv2.erode(horizontal, horizontalStructure)
    #     horizontal = cv2.dilate(horizontal, horizontalStructure)
    #
    #     horizontal = cv2.dilate(horizontal, (1, 1), iterations=5)
    #     horizontal = cv2.erode(horizontal, (1, 1), iterations=3)
    #
    #     ## Uncomment to visualize highlighted Horizontal lines
    #     # cv2.imshow("horizontal",horizontal)
    #     # cv2.waitKey(0)
    #
    #     # HoughlinesP function to detect horizontal lines
    #     hor_lines = cv2.HoughLinesP(horizontal, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=3)
    #     if hor_lines is None:
    #         return None, None
    #     temp_line = []
    #     for line in hor_lines:
    #         for x1, y1, x2, y2 in line:
    #             temp_line.append([x1, y1 - 5, x2, y2 - 5])
    #
    #     # Sorting the list of detected lines by Y1
    #     hor_lines = sorted(temp_line, key=lambda x: x[1])
    #
    #     ## Uncomment this part to visualize the lines detected on the image ##
    #     # print(len(hor_lines))
    #     # for x1, y1, x2, y2 in hor_lines:
    #     #     cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)
    #
    #     # print(image.shape)
    #     # cv2.imshow("image",image)
    #     # cv2.waitKey(0)
    #     ####################################################################
    #
    #     ## Selection of best lines from all the horizontal lines detected ##
    #     lasty1 = -111111
    #     lines_x1 = []
    #     lines_x2 = []
    #     hor = []
    #     i = 0
    #     for x1, y1, x2, y2 in hor_lines:
    #         if y1 >= lasty1 and y1 <= lasty1 + 10:
    #             lines_x1.append(x1)
    #             lines_x2.append(x2)
    #         else:
    #             if (i != 0 and len(lines_x1) is not 0):
    #                 hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
    #             lasty1 = y1
    #             lines_x1 = []
    #             lines_x2 = []
    #             lines_x1.append(x1)
    #             lines_x2.append(x2)
    #             i += 1
    #     hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
    #     #####################################################################
    #
    #     # [vertical lines]
    #     # Create structure element for extracting vertical lines through morphology operations
    #     verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 29))
    #
    #     # Apply morphology operations
    #     vertical = cv2.erode(vertical, verticalStructure)
    #     vertical = cv2.dilate(vertical, verticalStructure)
    #
    #     vertical = cv2.dilate(vertical, (1, 1), iterations=9)
    #     vertical = cv2.erode(vertical, (1, 1), iterations=3)
    #
    #     ######## Preprocessing Vertical Lines ###############
    #     # cv2.imshow("vertical",vertical)
    #     # cv2.waitKey(0)
    #     #####################################################
    #
    #     # HoughlinesP function to detect vertical lines
    #     # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    #     ver_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 20, np.array([]), 20, 2)
    #     if ver_lines is None:
    #         return None, None
    #     temp_line = []
    #     for line in ver_lines:
    #         for x1, y1, x2, y2 in line:
    #             temp_line.append([x1, y1, x2, y2])
    #
    #     # Sorting the list of detected lines by X1
    #     ver_lines = sorted(temp_line, key=lambda x: x[0])
    #
    #     ## Uncomment this part to visualize the lines detected on the image ##
    #     # print(len(ver_lines))
    #     # for x1, y1, x2, y2 in ver_lines:
    #     #     cv2.line(image, (x1,y1-5), (x2,y2-5), (0, 255, 0), 1)
    #
    #     # print(image.shape)
    #     # cv2.imshow("image",image)
    #     # cv2.waitKey(0)
    #     ####################################################################
    #
    #     ## Selection of best lines from all the vertical lines detected ##
    #     lastx1 = -111111
    #     lines_y1 = []
    #     lines_y2 = []
    #     ver = []
    #     count = 0
    #     lasty1 = -11111
    #     lasty2 = -11111
    #     for x1, y1, x2, y2 in ver_lines:
    #         if x1 >= lastx1 and x1 <= lastx1 + 15 and not (
    #                 ((min(y1, y2) < min(lasty1, lasty2) - 20 or min(y1, y2) < min(lasty1, lasty2) + 20)) and (
    #         (max(y1, y2) < max(lasty1, lasty2) - 20 or max(y1, y2) < max(lasty1, lasty2) + 20))):
    #             lines_y1.append(y1)
    #             lines_y2.append(y2)
    #             # lasty1 = y1
    #             # lasty2 = y2
    #         else:
    #             if (count != 0 and len(lines_y1) is not 0):
    #                 ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
    #             lastx1 = x1
    #             lines_y1 = []
    #             lines_y2 = []
    #             lines_y1.append(y1)
    #             lines_y2.append(y2)
    #             count += 1
    #             lasty1 = -11111
    #             lasty2 = -11111
    #     ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
    #     #################################################################
    #
    #     ############ Visualization of Lines After Post Processing ############
    #     # for x1, y1, x2, y2 in ver:
    #     #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
    #
    #     # for x1, y1, x2, y2 in hor:
    #     #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
    #
    #     # cv2.imshow("image",img)
    #     # cv2.waitKey(0)
    #     #######################################################################
    #
    #     return hor, ver

    def line_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1
        )
        bw = cv2.bitwise_not(bw)

        horizontal = bw.copy()
        vertical = bw.copy()

        # [horizontal lines]
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 1))

        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        horizontal = cv2.dilate(
            horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3
        )
        # horizontal = cv2.erode(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        # horizontal = cv2.dilate(horizontal, (1, 1), iterations=5)
        # horizontal = cv2.erode(horizontal, (1, 1), iterations=3)
        # cv2.imshow('horizontal', horizontal)
        # cv2.waitKey()

        hor_lines = cv2.HoughLinesP(
            horizontal,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=20,
            maxLineGap=3,
        )

        if hor_lines is None:
            return None, None
        temp_line = []
        for line in hor_lines:
            for x1, y1, x2, y2 in line:
                temp_line.append([x1, y1 - 5, x2, y2 - 5])

        hor_lines = sorted(temp_line, key=lambda x: x[1])

        # Selection of best lines from all the horizontal lines detected
        lasty1 = -111111
        lines_x1 = []
        lines_x2 = []
        hor = []
        i = 0
        for x1, y1, x2, y2 in hor_lines:
            if y1 >= lasty1 and y1 <= lasty1 + 10:
                lines_x1.append(x1)
                lines_x2.append(x2)
            else:
                if i != 0 and len(lines_x1) != 0:
                    hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
                lasty1 = y1
                lines_x1 = []
                lines_x2 = []
                lines_x1.append(x1)
                lines_x2.append(x2)
                i += 1
        hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])

        # [vertical lines]
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 29))

        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        vertical = cv2.dilate(
            vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3
        )
        # vertical = cv2.erode(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        # vertical = cv2.dilate(vertical, (1, 1), iterations=9)
        # vertical = cv2.erode(vertical, (1, 1), iterations=3)
        # cv2.imwrite('./buffer/vv.jpg', vertical)
        # cv2.imshow('vertical', vertical)
        # cv2.waitKey()

        ver_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 20, np.array([]), 20, 2)

        if ver_lines is None:
            return None, None
        temp_line = []
        for line in ver_lines:
            for x1, y1, x2, y2 in line:
                temp_line.append([x1, y1, x2, y2])

        # Sorting the list of detected lines by X1
        ver_lines = sorted(temp_line, key=lambda x: x[0])

        ## Selection of best lines from all the vertical lines detected ##
        lastx1 = -111111
        lines_y1 = []
        lines_y2 = []
        ver = []
        count = 0
        lasty1 = -11111
        lasty2 = -11111
        for x1, y1, x2, y2 in ver_lines:
            if (
                x1 >= lastx1
                and x1 <= lastx1 + 15
                and not (
                    (
                        (
                            min(y1, y2) < min(lasty1, lasty2) - 20
                            or min(y1, y2) < min(lasty1, lasty2) + 20
                        )
                    )
                    and (
                        (
                            max(y1, y2) < max(lasty1, lasty2) - 20
                            or max(y1, y2) < max(lasty1, lasty2) + 20
                        )
                    )
                )
            ):
                lines_y1.append(y1)
                lines_y2.append(y2)
                # lasty1 = y1
                # lasty2 = y2
            else:
                if count != 0 and len(lines_y1) != 0:
                    ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
                lastx1 = x1
                lines_y1 = []
                lines_y2 = []
                lines_y1.append(y1)
                lines_y2.append(y2)
                count += 1
                lasty1 = -11111
                lasty2 = -11111
        ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])

        return hor, ver

    def get_flag_line_noline_cls(self, image, boxes, radio_255_0=0.8, radio_gap=0.0067):
        img_h, img_w, _ = np.shape(image)
        temp_lines_hor, temp_lines_ver = self.line_detection(image)
        if temp_lines_hor is None:
            temp_lines_hor = []
        if temp_lines_ver is None:
            temp_lines_ver = []
        temp_lines_hor.append([0, 0, img_w, 0])
        temp_lines_hor.append([0, img_h, img_w, img_h])
        temp_lines_ver.append([0, 0, 0, img_h])
        temp_lines_ver.append([img_w, 0, img_w, img_h])

        temp = []
        sort_list = []
        for line in temp_lines_hor:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) / img_w > 0.5:
                temp.append([x1, y1, x2, y2])
                sort_list.append(y1)
        temp_lines_hor = np.array(temp)[
            sorted(range(len(sort_list)), key=lambda k: sort_list[k])
        ]

        temp = []
        sort_list = []
        for line in temp_lines_ver:
            x1, y1, x2, y2 = line
            if abs(y1 - y2) / img_h > 0.5:
                temp.append([x1, y1, x2, y2])
                sort_list.append(x1)
        temp_lines_ver = np.array(temp)[
            sorted(range(len(sort_list)), key=lambda k: sort_list[k])
        ]

        # search line show start(不用注释掉)
        # show_image = image.copy()
        # for line in temp_lines_hor:
        #     cv2.line(show_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
        # for line in temp_lines_ver:
        #     cv2.line(show_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
        # cv2.imwrite('./buffer/ioio.jpg', show_image)
        # search line show end

        h_set_compress, v_set_compress = self.get_gap(
            image, boxes, radio_255_0, radio_gap
        )
        h_set_compress_average = np.average(
            [
                abs(h_set_compress[index] - h_set_compress[index - 1])
                for index in range(1, len(h_set_compress))
            ]
        )
        v_set_compress_average = np.average(
            [
                abs(v_set_compress[index] - v_set_compress[index - 1])
                for index in range(1, len(v_set_compress))
            ]
        )

        for index in range(1, len(temp_lines_hor)):
            if (
                abs(temp_lines_hor[index][1] - temp_lines_hor[index - 1][1])
                > 5 * h_set_compress_average
            ):
                return 1

        for index in range(1, len(temp_lines_ver)):
            if (
                abs(temp_lines_ver[index][0] - temp_lines_ver[index - 1][0])
                > 3 * v_set_compress_average
            ):
                return 1
        return 0  # 0代表有线表格，1代表无线表格


if __name__ == "__main__":
    # base_path = '/Volumes/my_disk/company/sensedeal/buffer_disk/buffer_7/'
    base_path = "/Volumes/my_disk/company/sensedeal/217_PycharmProject/bbtv/yolov3_table_cells/runs/detect/exp5/"
    image_name_list = os.listdir(base_path)
    for image_name in image_name_list:
        main_start_time = time.time()
        # 准备好图片和boxes
        if image_name.startswith("show_"):
            continue
        if image_name.endswith(".txt"):
            continue
        if not os.path.exists(base_path + "".join(image_name.split(".")[:-1]) + ".txt"):
            continue
        if image_name != "14-1.jpg":
            continue
        print("image name: ", image_name)
        image_path = base_path + image_name
        label_path = base_path + "".join(image_name.split(".")[:-1]) + ".txt"
        ori_image = cv2.imread(image_path)
        ori_boxes = []
        for box in open(label_path, "r").readlines():
            box = [int(i) for i in box.rstrip("\n").split(" ")]
            ori_boxes.append(box)

        # 定义merge类
        merge_cls = merge_line_cell()

        # 具体操作，流程是融合线→融合box→融合线→转换成excel存储
        merge_h_line_list_, merge_v_line_list_, merge_boxes = merge_cls.merge_line(
            ori_image.copy(),
            copy.deepcopy(ori_boxes),
            iou_threshold=-0.1,
            distance_threshold="calculate",
            show_flag=False,
        )  # 0.025
        fill_boxes = merge_cls.fill_table(copy.deepcopy(merge_boxes))
        _, _, fill_boxes = merge_cls.merge_line(
            ori_image.copy(),
            copy.deepcopy(fill_boxes),
            iou_threshold=-0.1,
            distance_threshold="calculate",
            show_flag=False,
        )
        merge_cls.table_structure(
            "/Volumes/my_disk/company/sensedeal/buffer_disk/buffer_8/1.xlsx",
            copy.deepcopy(fill_boxes),
        )

        # 显示图片，演示效果
        print("per image use time: ", time.time() - main_start_time)
        fill_image = ori_image.copy()
        for box in fill_boxes:
            cv2.rectangle(
                fill_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2
            )
        cv2.imshow(
            "fill_image",
            cv2.resize(
                fill_image, (np.shape(fill_image)[1] // 1, np.shape(fill_image)[0] // 1)
            ),
        )
        cv2.waitKey()
