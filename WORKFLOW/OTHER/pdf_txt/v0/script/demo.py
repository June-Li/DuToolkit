import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import yaml

from MODELALG.utils.common import Log

logger = Log(__name__).get_logger()
logger.info("\n" * 2 + "*" * 25 + " OCR START " + "*" * 25)

from WORKFLOW.DET.SealDet.v2.detector_seal import Detector as Detector_seal
from WORKFLOW.RES.DeSeal.v0.de_seal import De as De_seal
from WORKFLOW.OTHER.DeSealProcess.v0.seal_process import SealProcessOperator
from WORKFLOW.OTHER.TableStruct.v4.table_struct import TableStructOperator
from WORKFLOW.DET.TextDet.v0.detector_text import Detector as Detector_text
from WORKFLOW.CLS.TextCls.v0.classifier_text import Classifier as Classifier_text
from WORKFLOW.REC.TextRec.v0.recognizer_text import Recognizer as Recognizer_text
from WORKFLOW.DET.TableDet.v1.detector_table import Detector as Detector_table
from WORKFLOW.CLS.TableCls.v0.classifier_table import Classifier as Classifier_table
from WORKFLOW.DET.TableCellDet.v1.detector_table_cell import (
    Detector as Detector_table_cell,
)
from WORKFLOW.DET.TableCellContentDet.v1.detector_table_cell_content import (
    Detector as Detector_table_cell_content,
)
from pdf2txt import Pdf2Txt


class TextSystem(object):
    def __init__(self, configs):
        logger.info("device is: " + configs["model"]["Detector_seal"]["device"])
        self.seal_detector = Detector_seal(
            model_path=configs["model"]["Detector_seal"]["model_path"],
            img_size=configs["model"]["Detector_seal"]["img_size"],
            conf_thres=configs["model"]["Detector_seal"]["conf_thres"],
            iou_thres=configs["model"]["Detector_seal"]["iou_thres"],
            device=configs["model"]["Detector_seal"]["device"],
            half_flag=configs["model"]["Detector_seal"]["model_path"],
        )
        self.seal_eliminator = De_seal(
            model_path=configs["model"]["De_seal"]["model_path"],
            img_size=configs["model"]["De_seal"]["img_size"],
            batch_size=configs["model"]["De_seal"]["batch_size"],
            device=configs["model"]["De_seal"]["device"],
            half_flag=configs["model"]["De_seal"]["half_flag"],
        )
        self.text_detector = Detector_text(
            model_path=configs["model"]["Detector_text"]["model_path"],
            post_process_num_works=configs["model"]["Detector_text"][
                "post_process_num_works"
            ],
            device=configs["model"]["Detector_text"]["device"],
            half_flag=False,  # configs['model']['Detector_text']['half_flag']
        )
        self.text_classifier = Classifier_text(
            net=configs["model"]["Classifier_text"]["net"],
            model_path=configs["model"]["Classifier_text"]["model_path"],
            device=configs["model"]["Classifier_text"]["device"],
            half_flag=configs["model"]["Classifier_text"]["half_flag"],
        )
        self.text_recognizer = Recognizer_text(
            model_path=configs["model"]["Recognizer_text"]["model_path"],
            device=configs["model"]["Recognizer_text"]["device"],
            half_flag=configs["model"]["Recognizer_text"]["half_flag"],
        )
        self.table = Detector_table(
            model_path=configs["model"]["Detector_table"]["model_path"],
            img_size=configs["model"]["Detector_table"]["img_size"],
            conf_thres=configs["model"]["Detector_table"]["conf_thres"],
            iou_thres=configs["model"]["Detector_table"]["iou_thres"],
            device=configs["model"]["Detector_table"]["device"],
            half_flag=configs["model"]["Detector_table"]["half_flag"],
        )
        self.table_classifier = Classifier_table(
            net=configs["model"]["Classifier_table"]["net"],
            model_path=configs["model"]["Classifier_table"]["model_path"],
            device=configs["model"]["Classifier_table"]["device"],
            half_flag=configs["model"]["Classifier_table"]["half_flag"],
        )
        self.table_cell = Detector_table_cell(
            model_path=configs["model"]["Detector_table_cell"]["model_path"],
            img_size=configs["model"]["Detector_table_cell"]["img_size"],
            conf_thres=configs["model"]["Detector_table_cell"]["conf_thres"],
            iou_thres=configs["model"]["Detector_table_cell"]["iou_thres"],
            device=configs["model"]["Detector_table_cell"]["device"],
            half_flag=configs["model"]["Detector_table_cell"]["half_flag"],
        )
        self.table_cell_content = Detector_table_cell_content(
            model_path=configs["model"]["Detector_table_cell_content"]["model_path"],
            img_size=configs["model"]["Detector_table_cell_content"]["img_size"],
            conf_thres=configs["model"]["Detector_table_cell_content"]["conf_thres"],
            iou_thres=configs["model"]["Detector_table_cell_content"]["iou_thres"],
            device=configs["model"]["Detector_table_cell_content"]["device"],
            half_flag=configs["model"]["Detector_table_cell_content"]["half_flag"],
        )
        logger.info(" ···-> all load model succeeded!")
        self.seal_process = SealProcessOperator()
        self.table_struct = TableStructOperator()


configs = yaml.load(open(root_dir + "/system.yml", "r"), Loader=yaml.FullLoader)
configs["scan_flag"] = True
text_sys = TextSystem(configs)

file_converter = Pdf2Txt(text_sys, configs)

file_path = "/workspace/JuneLi/bbtv/SensedealImgAlg/WORKFLOW/OTHER/OCR/v3/test_data/fake/pdf_fake.pdf"
hypertxt = file_converter.apply_one(file_path)
print(hypertxt)
