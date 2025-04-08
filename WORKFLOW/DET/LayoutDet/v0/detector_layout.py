import onnxruntime as ort
import math
import numpy as np
import cv2

from MODELALG.utils.common import Log


logger = Log(__name__).get_logger()


class Detector(object):
    def __init__(self, model_path=None, thr=0.7, device="0"):
        self.thr = thr
        self.device = device
        if self.device.lower() != "cpu" and ort.get_device() == "GPU":
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            self.ort_sess = ort.InferenceSession(
                model_path, options=options, providers=[("CUDAExecutionProvider")]
            )
        else:
            self.ort_sess = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        self.input_names = [node.name for node in self.ort_sess.get_inputs()]
        self.output_names = [node.name for node in self.ort_sess.get_outputs()]
        self.input_shape = self.ort_sess.get_inputs()[0].shape[2:4]
        self.label_list = [
            "_background_",
            "Text",
            "Title",
            "Figure",
            "Figure caption",
            "Table",
            "Table caption",
            "Header",
            "Footer",
            "Reference",
            "Equation",
        ]

    @staticmethod
    def decode_image(im_file, im_info):
        """read rgb image
        Args:
            im_file (str|np.ndarray): input can be image path or np.ndarray
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        if isinstance(im_file, str):
            with open(im_file, "rb") as f:
                im_read = f.read()
            data = np.frombuffer(im_read, dtype="uint8")
            im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = im_file
        im_info["im_shape"] = np.array(im.shape[:2], dtype=np.float32)
        im_info["scale_factor"] = np.array([1.0, 1.0], dtype=np.float32)
        return im, im_info

    @staticmethod
    def operators_preprocess(im, preprocess_ops):
        # process image by preprocess_ops
        im_info = {
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
            "im_shape": None,
        }
        im, im_info = Detector.decode_image(im, im_info)
        for operator in preprocess_ops:
            im, im_info = operator(im, im_info)
        return im, im_info

    def preprocess(self, image_list):
        inputs = []
        if "scale_factor" in self.input_names:
            preprocess_ops = []
            for op_info in [
                {
                    "interp": 2,
                    "keep_ratio": False,
                    "target_size": [800, 608],
                    "type": "LinearResize",
                },
                {
                    "is_scale": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "type": "StandardizeImage",
                },
                {"type": "Permute"},
                {"stride": 32, "type": "PadStride"},
            ]:
                new_op_info = op_info.copy()
                op_type = new_op_info.pop("type")
                preprocess_ops.append(eval(op_type)(**new_op_info))

            for im_path in image_list:
                im, im_info = Detector.preprocess(im_path, preprocess_ops)
                inputs.append(
                    {
                        "image": np.array((im,)).astype("float32"),
                        "scale_factor": np.array((im_info["scale_factor"],)).astype(
                            "float32"
                        ),
                    }
                )
        else:
            hh, ww = self.input_shape
            for img in image_list:
                h, w = img.shape[:2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(np.array(img).astype("float32"), (ww, hh))
                # Scale input pixel values to 0 to 1
                img /= 255.0
                img = img.transpose(2, 0, 1)
                img = img[np.newaxis, :, :, :].astype(np.float32)
                inputs.append(
                    {self.input_names[0]: img, "scale_factor": [w / ww, h / hh]}
                )
        return inputs

    def postprocess(self, boxes, inputs, thr):
        if "scale_factor" in self.input_names:
            bb = []
            for b in boxes:
                clsid, bbox, score = int(b[0]), b[2:], b[1]
                if score < thr:
                    continue
                if clsid >= len(self.label_list):
                    logger.warning(f"bad category id")
                    continue
                bb.append(
                    {
                        "type": self.label_list[clsid].lower(),
                        "bbox": [int(t) for t in bbox.tolist()],
                        "score": float(score),
                    }
                )
            return bb

        def xywh2xyxy(x):
            # [x, y, w, h] to [x1, y1, x2, y2]
            y = np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        def compute_iou(box, boxes):
            # Compute xmin, ymin, xmax, ymax for both boxes
            xmin = np.maximum(box[0], boxes[:, 0])
            ymin = np.maximum(box[1], boxes[:, 1])
            xmax = np.minimum(box[2], boxes[:, 2])
            ymax = np.minimum(box[3], boxes[:, 3])

            # Compute intersection area
            intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

            # Compute union area
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union_area = box_area + boxes_area - intersection_area

            # Compute IoU
            iou = intersection_area / union_area

            return iou

        def iou_filter(boxes, scores, iou_threshold):
            sorted_indices = np.argsort(scores)[::-1]

            keep_boxes = []
            while sorted_indices.size > 0:
                # Pick the last box
                box_id = sorted_indices[0]
                keep_boxes.append(box_id)

                # Compute IoU of the picked box with the rest
                ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

                # Remove boxes with IoU over the threshold
                keep_indices = np.where(ious < iou_threshold)[0]

                # print(keep_indices.shape, sorted_indices.shape)
                sorted_indices = sorted_indices[keep_indices + 1]

            return keep_boxes

        boxes = np.squeeze(boxes).T
        # Filter out object confidence scores below threshold
        scores = np.max(boxes[:, 4:], axis=1)
        boxes = boxes[scores > thr, :]
        scores = scores[scores > thr]
        if len(boxes) == 0:
            return []

        # Get the class with the highest confidence
        class_ids = np.argmax(boxes[:, 4:], axis=1)
        boxes = boxes[:, :4]
        input_shape = np.array(
            [
                inputs["scale_factor"][0],
                inputs["scale_factor"][1],
                inputs["scale_factor"][0],
                inputs["scale_factor"][1],
            ]
        )
        boxes = np.multiply(boxes, input_shape, dtype=np.float32)
        boxes = xywh2xyxy(boxes)

        unique_class_ids = np.unique(class_ids)
        indices = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = iou_filter(class_boxes, class_scores, 0.2)
            indices.extend(class_indices[class_keep_boxes])

        return [
            {
                "type": self.label_list[class_ids[i]].lower(),
                "bbox": [int(t) for t in boxes[i].tolist()],
                "score": float(scores[i]),
            }
            for i in indices
        ]

    def __call__(self, image_list, batch_size=16):
        res = []
        imgs = []
        for i in range(len(image_list)):
            if not isinstance(image_list[i], np.ndarray):
                imgs.append(np.array(image_list[i]))
            else:
                imgs.append(image_list[i])

        batch_loop_cnt = math.ceil(float(len(imgs)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(imgs))
            batch_image_list = imgs[start_index:end_index]
            inputs = self.preprocess(batch_image_list)
            for ins in inputs:
                bb = self.postprocess(
                    self.ort_sess.run(
                        None, {k: v for k, v in ins.items() if k in self.input_names}
                    )[0],
                    ins,
                    self.thr,
                )
                res.append(bb)
        return res


if __name__ == "__main__":
    import os
    import fitz
    from PIL import Image

    def pdf2img(pdf_path, max_len=3500):
        doc = fitz.open(pdf_path)
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = np.shape(img)
            if max(h, w) > max_len:
                if h > w:
                    img = cv2.resize(img, (max_len * w // h, max_len))
                else:
                    img = cv2.resize(img, (max_len, max_len * h // w))
            yield img

    # def pdf2img(pdf_path, zoomin=3):
    #     images = []
    #     pdf = fitz.open(pdf_path)
    #     mat = fitz.Matrix(zoomin, zoomin)
    #     for i, page in enumerate(pdf):
    #         pix = page.get_pixmap(matrix=mat)
    #         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #         images.append(img)
    #     return images

    op = Detector("/volume/weights/layout.onnx")
    # imgs = pdf2img("/volume/test_data/fake/pdf_fake_text_table.pdf")
    imgs = pdf2img(
        "/volume/test_data/多场景数据测试/ESG/20210601-中信证券-全球产业投资2021年下半年投资策略：ESG和AI引领下的全球产业投资策略.pdf"
    )
    # res = op(list(imgs))
    for idx, img in enumerate(imgs):
        res = op([img])
        for elem in res[0]:
            bbox = elem["bbox"]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        print()
