# Copyright (c) SDAI. All rights reserved.
import json
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../")
sys.path.append(os.path.abspath(root_dir))

import base64
import copy
import datetime
import re
import time
from datetime import timedelta

import cv2
import fitz
import gradio as gr
import numpy as np
from gradio_pdf import PDF
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from MODELALG.utils import common
from WORKFLOW.OTHER.OCR.v0.OCRModelDeployMulti import (
    BorderlessTableAlg,
    BorderTableAlg,
    ChapterAlg,
    LayoutAlg,
    OcrOperator,
    OCRType,
    Resolution,
)

op = OcrOperator()

os.environ["GRADIO_TEMP_DIR"] = cur_dir + "/upload"


def clean_upload_directory(directory_path, max_age_hours=24):
    """
    清理上传目录中超过指定时间的文件

    Args:
        directory_path: 要清理的目录路径
        max_age_hours: 文件保留的最大时间（小时）
    """
    try:
        if not os.path.exists(directory_path):
            return

        current_time = datetime.datetime.now()
        max_age = timedelta(hours=max_age_hours)

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_creation_time = datetime.datetime.fromtimestamp(
                    os.path.getctime(file_path)
                )
                if current_time - file_creation_time > max_age:
                    os.remove(file_path)
                    logger.info(f"已删除过期文件: {file_path}")
            elif os.path.isdir(file_path):
                # 递归清理子目录
                clean_upload_directory(file_path, max_age_hours)
                # 如果目录为空，则删除
                if not os.listdir(file_path):
                    os.rmdir(file_path)
    except Exception as e:
        logger.error(f"清理上传目录时出错: {str(e)}")


def update_preview(file_obj):
    if file_obj is None:
        return None, None, gr.update(visible=True), gr.update(visible=False)

    file_path = file_obj.name
    file_ext = file_path.lower().split(".")[-1] if "." in file_path else ""

    if file_ext.lower() in ["pdf"]:
        # PDF文件
        return file_path, None, gr.update(visible=True), gr.update(visible=False)
    elif file_ext.lower() in ["png", "jpg", "jpeg"]:
        # 图片文件
        return None, file_obj, gr.update(visible=False), gr.update(visible=True)
    else:
        # 不支持的文件类型
        return None, None, gr.update(visible=False), gr.update(visible=False)


def convert_to_html_table(data):
    # 写入html开始
    max_row = max([elem[0][2] for elem in data])
    max_col = max([elem[0][3] for elem in data])
    html_table_list = []
    for i in range(max_row):
        html_h_list = []
        for j in range(max_col):
            html_h_list.append("")
        html_table_list.append(html_h_list)

    index = 0
    for elem in data:
        y_0, x_0, y_1, x_1 = elem[0]
        cell_str = "<td "
        cell_str = cell_str + "class=" + '"' + "tg-0lax" + '" '
        cell_str = (
            cell_str + "rowspan=" + '"' + str(y_1 - y_0) + '" '
        )  # 向下融合cell的数量
        cell_str = (
            cell_str + "colspan=" + '"' + str(x_1 - x_0) + '" '
        )  # 向右融合cell的数量
        # cell_str = cell_str + "height=" + '"' + str(box[3] - box[1]) + '" '  # 设置cell的宽
        # cell_str = cell_str + "width=" + '"' + str(box[2] - box[0]) + '" '  # 设置cell的高
        cell_str = cell_str + ">"
        cell_str = cell_str + elem[1]  # 文本内容
        cell_str = cell_str + "</td>"  # 结束符
        html_table_list[y_0][x_0] = cell_str
        index += 1
    html_str = "<table>\n"
    for i in html_table_list:
        if i == [""] * len(i):
            continue
        html_str += "<tr>\n"
        for j in i:
            if j != "":
                html_str += j + "\n"
        html_str += "</tr>\n"
    html_str += "</table>\n"
    return html_str


def latex_to_markdown(latex_str):
    # 处理数学函数格式
    latex_str = re.sub(
        r"\\mathrm\s+([A-Za-z ]+)(?![^{}]*})", r"\\mathrm{\1}", latex_str
    )
    latex_str = re.sub(
        r"\\mathrm{([^}]*)}",
        lambda m: f'\\mathrm{{{m.group(1).replace(" ", "")}}}',
        latex_str,
    )
    latex_str = re.sub(r"\\mathrm{([^}]*)}(\s*)(\d+)", r"\\mathrm{\1\3}", latex_str)

    # 修复关键点：精准匹配括号对
    # 新增：只处理成对括号内的空格
    latex_str = re.sub(
        r"$\s*([^)]+?)\s*$", r"(\1)", latex_str
    )  # 匹配 ( content ) → (content)

    # 其他优化保持不变
    latex_str = re.sub(
        r"_{([^}]*)}", lambda m: "_{" + m.group(1).replace(" ", "") + "}", latex_str
    )
    latex_str = re.sub(r",\s+", ",", latex_str)
    latex_str = re.sub(r"\s*-\s*", "-", latex_str)

    # 新增：保护公式末尾的标点符号
    return (
        f'${latex_str.strip().rstrip(").")}.)$'
        if latex_str.strip().endswith(".)")
        else f"${latex_str.strip()}$"
    )


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def hypertxt2markdown(hypertxt, file_path):
    def page_to_img(page):
        max_len = 3500
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = np.shape(img)
        if max(h, w) > max_len:
            if h > w:
                img = cv2.resize(img, (max_len * w // h, max_len))
            else:
                img = cv2.resize(img, (max_len, max_len * h // w))
        return img

    def replace_angle_brackets_advanced(text):
        """
        处理可能的嵌套尖括号和更复杂的情况
        """
        # 非贪婪匹配，找到最小的<...>对
        pattern = r"<(.*?)>"

        def replacer(match):
            # 获取<>之间的内容
            inner_content = match.group(1)
            # 返回替换后的格式
            return f"&lt;{inner_content}&gt;"

        return re.sub(pattern, replacer, text)

    rejust_imgs = {}
    doc = fitz.open(file_path)

    pages_bbox = hypertxt["metadata"]["pages_bbox"]
    mdstr = ""
    for block in hypertxt["context"]:
        if block["type"] == "text":
            if block["is_title"]:
                cid_len = len(block["cid"].split("."))
                mdstr += "#" * cid_len + " " + block["text"] + "\n"
            else:
                mdstr += replace_angle_brackets_advanced(block["text"]) + "\n"
        elif block["type"] == "table":
            mdstr += convert_to_html_table(block["text"]) + "\n"
        elif block["type"] == "formula":
            starttime = time.time()
            mdstr += latex_to_markdown(block["text"]) + "\n"
            print("latex to md use time:", time.time() - starttime)
        elif block["type"] == "figure":
            page_idx = block["page_idx"] - 1
            if page_idx not in rejust_imgs.keys():
                rj_img = page_to_img(doc[page_idx])
                rj_img = common.warp_affine_img(
                    rj_img.copy(), hypertxt["metadata"]["rot_angle"][page_idx]
                )
                rj_img = cv2.resize(
                    rj_img, (pages_bbox[page_idx][2], pages_bbox[page_idx][3])
                )
                rejust_imgs[page_idx] = rj_img
            rj_img = rejust_imgs[page_idx]
            block_box = block["figure_box"]
            figure_img = copy.deepcopy(
                rj_img[block_box[1] : block_box[3], block_box[0] : block_box[2]]
            )
            success, buffer = cv2.imencode(
                ".jpg", figure_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            if not success:
                buffer = b""
            base64_image = base64.b64encode(buffer).decode("utf-8")
            # mdstr += f"![](data:image/jpeg;base64,{base64_image})\n"
            description = block["text"]["vl_result"].replace("\n", " ")
            mdstr += f'![图片加载失败](data:image/jpeg;base64,{base64_image} "{description}")\n'
    return mdstr


def to_markdown(
    file_path,
    force_cv=OCRType.YES,
    resolution=Resolution.HIGH,
    de_seal_enable=True,
    fix_text_enable=True,
    char_box_enable=True,
    tilt_correction_enable=True,
    border_table_alg=BorderTableAlg.SDBORDERTABLE_CV_MODEL,
    borderless_table_alg=BorderlessTableAlg.SDBORDERLESSTABLE_LM_LITE_MODEL,
    layout_alg=LayoutAlg.SDLAYOUT,
    formula_enable=True,
    figure_rec_by_vl_enbale=False,
    figure_upload_enable=False,
    chapter_alg=ChapterAlg.RE,
    reading_order_enable=True,
):
    try:
        starttime = time.time()
        hypertxt = op(
            file_path,
            force_cv=force_cv,
            resolution=resolution,
            de_seal_enable=de_seal_enable,
            fix_text_enable=fix_text_enable,
            char_box_enable=char_box_enable,
            tilt_correction_enable=tilt_correction_enable,
            border_table_alg=border_table_alg,
            borderless_table_alg=borderless_table_alg,
            layout_alg=layout_alg,
            formula_enable=formula_enable,
            figure_rec_by_vl_enbale=figure_rec_by_vl_enbale,
            figure_upload_enable=figure_upload_enable,
            chapter_alg=chapter_alg,
            reading_order_enable=reading_order_enable,
        )
        hypertxt = json.loads(hypertxt)
        consume_time = round(time.time() - starttime, 3)
        print("time:", consume_time)
        markdown_str = hypertxt2markdown(hypertxt, file_path)
        json_str = json.dumps(hypertxt, ensure_ascii=False, indent=4)

        # 处理完文件后，执行一次清理操作
        clean_upload_directory(
            cur_dir + "/upload", max_age_hours=24
        )  # 文件只保留24小时

        return (
            markdown_str,
            # json_str,
            "The total time consumed: " + str(consume_time) + "s!",
        )
    except Exception as e:
        logger.exception(e)
        clean_upload_directory(
            cur_dir + "/upload", max_age_hours=24
        )  # 文件只保留24小时
        return (
            "'Failed': " + str(e),
            # "{}",
            "Failed!",
        )


def apply_preset(preset):
    """根据预设配置更新所有参数"""
    if preset == "low":
        return {
            "force_cv": "yes",
            "resolution": "std",
            "formula_enable": False,
            "de_seal_enable": False,
            "fix_text_enable": False,
            "char_box_enable": False,
            "tilt_correction_enable": False,
            "reading_order_enable": False,
            "figure_rec_by_vl_enbale": False,
            "figure_upload_enable": False,
            "border_table_alg": "sdbordertable_cv_model",
            "borderless_table_alg": "sdborderlesstable_cv_lite_model",
            "layout_alg": "sdlayout",
            "chapter_alg": "re",
        }
    elif preset == "medium":
        return {
            "force_cv": "yes",
            "resolution": "high",
            "formula_enable": True,
            "de_seal_enable": True,
            "fix_text_enable": True,
            "char_box_enable": True,
            "tilt_correction_enable": True,
            "reading_order_enable": True,
            "figure_rec_by_vl_enbale": False,
            "figure_upload_enable": False,
            "border_table_alg": "sdbordertable_cv_model",
            "borderless_table_alg": "sdborderlesstable_cv_lite_model",
            "layout_alg": "sdlayout",
            "chapter_alg": "re",
        }
    elif preset == "high":
        return {
            "force_cv": "yes",
            "resolution": "high",
            "formula_enable": True,
            "de_seal_enable": True,
            "fix_text_enable": True,
            "char_box_enable": True,
            "tilt_correction_enable": True,
            "reading_order_enable": True,
            "figure_rec_by_vl_enbale": True,
            "figure_upload_enable": True,
            "border_table_alg": "sdbordertable_cv_model",
            "borderless_table_alg": "sdborderlesstable_cv_model",
            "layout_alg": "sdlayout",
            "chapter_alg": "re",
        }


if __name__ == "__main__":
    gradio_ip = "0.0.0.0"  # "192.168.1.80"
    gradio_port = 6061

    # 启动应用前先清理过期的上传文件
    clean_upload_directory(cur_dir + "/upload", max_age_hours=24)  # 文件只保留24小时

    with gr.Blocks() as demo:
        gr.HTML(open(cur_dir + "/header.html", "r", encoding="utf-8").read())
        with gr.Row():
            with gr.Column(variant="panel", scale=5):
                file = gr.File(
                    label="upload pdf or image",
                    file_types=[".pdf", ".png", ".jpeg", ".jpg"],
                )

                # 添加三个精度级别按钮
                with gr.Row():
                    low_preset_btn = gr.Button("低精度", variant="secondary")
                    medium_preset_btn = gr.Button("中精度", variant="primary")
                    high_preset_btn = gr.Button("高精度", variant="secondary")

                # 创建可折叠的高级选项区域
                with gr.Accordion("高级选项", open=False):
                    with gr.Row():
                        force_cv = gr.Dropdown(
                            label="force_cv",
                            choices=["auto", "yes", "no"],
                            value="yes",
                            interactive=True,
                        )
                        formula_enable = gr.Checkbox(
                            label="formula recognition enable", value=True
                        )
                        de_seal_enable = gr.Checkbox(
                            label="de seal enable", value=True, interactive=True
                        )
                        fix_text_enable = gr.Checkbox(
                            label="fix text enable", value=True, interactive=True
                        )
                        char_box_enable = gr.Checkbox(
                            label="char box enable", value=True, interactive=True
                        )
                        tilt_correction_enable = gr.Checkbox(
                            label="tilt correction enable", value=True, interactive=True
                        )
                        reading_order_enable = gr.Checkbox(
                            label="reading order enable", value=True
                        )
                        figure_rec_by_vl_enbale = gr.Checkbox(
                            label="figure recognition by vl enable", value=False
                        )
                        figure_upload_enable = gr.Checkbox(
                            label="figure upload enable", value=False
                        )
                    with gr.Row():
                        resolution = gr.Dropdown(
                            label="resolution",
                            choices=["high", "std"],
                            value="high",
                            interactive=True,
                        )
                        border_table_alg = gr.Dropdown(
                            label="border table alg",
                            choices=[
                                "sdbordertable_cv_model",
                            ],
                            value="sdbordertable_cv_model",
                            interactive=True,
                        )
                        borderless_table_alg = gr.Dropdown(
                            label="borderless table alg",
                            choices=[
                                "sdborderlesstable_cv_lite_model",
                                "sdborderlesstable_cv_model",
                                "sdborderlesstable_lm_lite_model",
                                "sdborderlesstable_lm_model",
                            ],
                            value="sdborderlesstable_lm_lite_model",
                            interactive=True,
                        )
                        layout_alg = gr.Dropdown(
                            label="layout alg",
                            choices=["sdlayout"],
                            value="sdlayout",
                            interactive=True,
                        )
                        chapter_alg = gr.Dropdown(
                            label="chapter recognition alg",
                            choices=["re", "layout", "llm"],
                            value="re",
                            interactive=True,
                        )
                with gr.Row():
                    time_label = gr.Label(label="processing time", value="")
                with gr.Row():
                    change_bu = gr.Button("开始处理", variant="primary")
                    clear_bu = gr.ClearButton(value="清除历史", variant="secondary")
                with gr.Row():
                    with gr.Group(visible=True) as preview_group:
                        pdf_show = PDF(
                            label="original pdf",
                            interactive=False,
                            visible=True,
                            height=1000,
                        )
                        img_show = gr.Image(
                            label="original image",
                            interactive=False,
                            visible=False,
                            height=1000,
                        )
                    with gr.Tabs():
                        with gr.Tab("MarkDown"):
                            md = gr.Markdown(
                                label="MarkDown",
                                height=1000,
                                show_copy_button=True,
                                latex_delimiters=[
                                    {"left": "$$", "right": "$$", "display": True},
                                    {"left": "$", "right": "$", "display": False},
                                ],
                                line_breaks=True,
                            )
                        # with gr.Tab("Json"):
                        #     json_info = gr.TextArea(lines=45, show_copy_button=True)

        # 连接预设按钮与参数更新
        all_params = [
            force_cv,
            resolution,
            formula_enable,
            de_seal_enable,
            fix_text_enable,
            char_box_enable,
            tilt_correction_enable,
            reading_order_enable,
            figure_rec_by_vl_enbale,
            figure_upload_enable,
            border_table_alg,
            borderless_table_alg,
            layout_alg,
            chapter_alg,
        ]

        def set_preset_values(preset):
            preset_values = apply_preset(preset)
            return [
                preset_values["force_cv"],
                preset_values["resolution"],
                preset_values["formula_enable"],
                preset_values["de_seal_enable"],
                preset_values["fix_text_enable"],
                preset_values["char_box_enable"],
                preset_values["tilt_correction_enable"],
                preset_values["reading_order_enable"],
                preset_values["figure_rec_by_vl_enbale"],
                preset_values["figure_upload_enable"],
                preset_values["border_table_alg"],
                preset_values["borderless_table_alg"],
                preset_values["layout_alg"],
                preset_values["chapter_alg"],
            ]

        def update_button_states(selected):
            """更新按钮状态，选中的按钮高亮显示"""
            low_state = gr.update(
                variant="primary" if selected == "low" else "secondary"
            )
            medium_state = gr.update(
                variant="primary" if selected == "medium" else "secondary"
            )
            high_state = gr.update(
                variant="primary" if selected == "high" else "secondary"
            )
            return [low_state, medium_state, high_state] + set_preset_values(selected)

        demo.load(
            fn=lambda: set_preset_values("medium"), inputs=None, outputs=all_params
        )

        # 更新按钮点击事件
        button_outputs = [
            low_preset_btn,
            medium_preset_btn,
            high_preset_btn,
        ] + all_params

        low_preset_btn.click(
            fn=lambda: update_button_states("low"), inputs=[], outputs=button_outputs
        )

        medium_preset_btn.click(
            fn=lambda: update_button_states("medium"), inputs=[], outputs=button_outputs
        )

        high_preset_btn.click(
            fn=lambda: update_button_states("high"), inputs=[], outputs=button_outputs
        )

        file.change(
            fn=update_preview,
            inputs=file,
            outputs=[
                pdf_show,  # PDF文件路径或None
                img_show,  # 图片文件或None
                pdf_show,  # PDF组件是否可见
                img_show,  # 图片组件是否可见
            ],
        )
        change_bu.click(
            fn=to_markdown,
            inputs=[
                file,
                force_cv,
                resolution,
                de_seal_enable,
                fix_text_enable,
                char_box_enable,
                tilt_correction_enable,
                border_table_alg,
                borderless_table_alg,
                layout_alg,
                formula_enable,
                figure_rec_by_vl_enbale,
                figure_upload_enable,
                chapter_alg,
                reading_order_enable,
            ],
            outputs=[md, time_label],
        )
        clear_bu.add([file, md, pdf_show, time_label])

    demo.queue(
        default_concurrency_limit=2,
        max_size=10,
        api_open=False,  # 可选：禁止直接通过API调用，强制走队列
    )
    demo.launch(
        server_name=gradio_ip,
        share=True,
        server_port=gradio_port,
    )
