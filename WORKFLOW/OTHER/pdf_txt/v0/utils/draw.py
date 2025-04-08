#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#                                                           
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved 
#    
# Description:
#   对pdfplumber提取结果进行可视化
#
# Author: Li Xiuming & Li Lin
# Last Modified: 2020-10-29                                                 
############################################################

import math
import random
from collections import namedtuple
import pdf2image
import pdfplumber as pp
from tqdm import tqdm
from PIL import Image, ImageDraw


class PdfDraw(object):
    def __init__(self, pdf_file):
        self.pdf = pp.open(pdf_file)
        self.pdf_images = pdf2image.convert_from_path(pdf_file)
        
    def draw_page_base(self, page_id, mode="chars", table=None):
        page = self.pdf.pages[page_id]
        if mode == "chars":
            objects = page.chars
        elif mode == "edges":
            objects = page.edges
        elif mode == "lines":
            objects = page.lines
        elif mode == "rects":
            objects = page.rects
        elif mode == "extract_words":
            objects = page.extract_words()
        elif mode == "extract_tables":
            objects = page.find_tables()
        elif mode == "table_cells" and table is not None:
            objects = table.cells
            
        page_height = page.height
        page_width = page.width
        bboxs = []
        BBOX = namedtuple("BBOX", ["x0", "top", "x1", "bottom"])
        
        for obj in objects:
            if obj["x1"] > page_width or obj["bottom"] > page_height:
                print('超出边界')
              
            if mode == "extract_tables" :
                bbox = BBOX._make(obj.bbox)
            elif mode == "table_cells":
                bbox = obj
            else:
                bbox = BBOX._make([obj["x0"], obj["top"], obj["x1"], obj["bottom"]])
                
            bboxs.append([bbox.x0, bbox.top, 
                          bbox.x1, bbox.top, 
                          bbox.x1, bbox.bottom, 
                          bbox.x0, bbox.bottom])
         
        self.draw_bboxs(bboxs, page_id)
        
        
    def draw_pages(self, mode="chars", page_ids="[ALL]"):
        if page_ids == "[ALL]":
            page_ids = range(len(self.pdf.pages))
            
        for page_id in tqdm(page_ids):
            self.draw_page_base(page_id, mode)
        

    def draw_bboxs(self, bbox, page_id):
        '''给某页的pdf着色'''
        this_page = self.pdf.pages[page_id]
        this_img = self.pdf_images[page_id].resize(
                (math.ceil(this_page.width), math.ceil(this_page.height)), Image.ANTIALIAS)
        new_img = this_img.copy()
        draw_img = ImageDraw.Draw(new_img)
        for box in bbox:
            # 生成随机颜色  若需固定颜色 可用（0,255,0）
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_img.polygon(box, fill=color)
        new_img = Image.blend(this_img, new_img, 0.5)
        new_img.show()
        #new_img.save("temp.jpg")


if __name__ == '__main__':

    pdf_file = "./test_data/临时报告（表格）3.pdf"
    pdf_draw = PdfDraw(pdf_file)
    pdf_draw.draw_pages(mode="rects", page_ids=[0, 1])

