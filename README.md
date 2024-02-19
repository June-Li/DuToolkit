## ToDo
- [ ] pdfplumber合并文字优化;  
- [ ] 单行拆增加gpt优化；
- [ ] 单行合并增加gpt优化；
- [ ] 有content-box后的结构化优化；

## 开发

使用自动格式化工具Black

PyCharm: 在 PyCharm -> Preferences 里找到 工具 -> Black，设置为保存时自动格式化。
VSCode: 使用这个插件，并且在设置-Text Editor-Formatting 里设置为保存时自动Format。

## Docker 封装
    到DEPLOY对应的CPU、GPU下运行sudo sh build.sh {version}即可
    

## 环境配置(目前用的)
    python：3.8.5
    cuda：11.0
    pytorch：1.7.1
    Driver Version: 440.82
    
## 运行时资源消耗
    GPU：
        CPU：2核
        卡数：单卡
        显存：8G(峰值)
        内存：5G
     CPU：
        CPU：6核
        内存：12G
    
## 结构介绍
        将整个图像算法进行解耦，分成模型算法、pt模型、业务流程、数据，业务模块可以任意选取其他业务模块或者模型算法进行组合，并且每个模型算法也是解耦的，可以选择不同的配置文件选择不同的b
    ackbone、neck、head、loss。    
        每个业务都写了API接口说明，对输入输出做了解释，可以对应到具体位置查看详细。
  
## 举例说明OCR业务
    OCR业务
        |→调用去除印章模块
            |→调用印章检测模块
                |→调用模型算法的检测模型
                    |→模型配置参数
            |→调用印章去除模块
                |→调用模型算法的图像修复模型
                    |→模型配置参数
        |→调用文本检测业务
            |→调用模型算法的检测模型
                |→模型配置参数
        |→调用文本方向分类业务
            |→调用模型算法的分类模型
                |→模型配置参数
        |→调用文本识别业务
            |→调用模型算法的识别模型
                |→模型配置参数
        |→调用表格结构化业务
            |→调用表格检测业务
                |→调用模型算法的检测模型
                    |→模型配置参数
            |→调用表格分类业务
                |→调用模型算法的分类模型
                    |→模型配置参数
            |→调用单元格检测业务
                |→调用模型算法的检测模型
                    |→模型配置参数
            |→调用单元格后处理业务
            
## 快速使用OCR（包括去除印章、文本检测识别、表格结构化）
    python OCRModelDeploy.py (SensedealImgAlg/WORKFLOW/OTHER/OCR/v3/OCRModelDeploy.py)
    
## 算法结构

        ├── DATASETS
    │   ├── CLS
    │   │   ├── TableCls
    │   │   │   └── v0
    │   │   └── TextCls
    │   │       └── v0
    │   ├── DET
    │   │   ├── SealDet
    │   │   │   ├── v0
    │   │   │   └── v1
    │   │   ├── TableCellDet
    │   │   │   ├── v0
    │   │   │   └── v1
    │   │   ├── TableDet
    │   │   │   └── v0
    │   │   └── TextDet
    │   │       ├── private
    │   │       └── public
    │   ├── REC
    │   │   └── TextRec
    │   │       └── icdar2015
    │   └── RES
    │       └── DeSeal
    │           ├── v0 -> /workspace/JuneLi/bbtv/Datasets/private/DeSeal/v0
    │           └── v1 -> /workspace/JuneLi/bbtv/Datasets/private/DeSeal/v1
    ├── MODEL
    │   ├── CLS
    │   │   └── ClsCollect
    │   │       └── ClsCollectv0
    │   ├── DET
    │   │   ├── DB
    │   │   │   └── DBv0
    │   │   └── YOLO
    │   │       ├── SSLYOLOv3
    │   │       ├── SSLYOLOv5
    │   │       ├── YOLOv3
    │   │       └── YOLOv5
    │   ├── REC
    │   │   └── CRNN
    │   │       └── CRNNv0
    │   └── RES
    │       └── MPRNet
    │           └── MPRNetv0
    ├── MODELALG
    │   ├── CLS
    │   │   └── ClsCollect
    │   │       └── ClsCollectv0
    │   ├── DET
    │   │   ├── DB
    │   │   │   └── DBv0
    │   │   └── YOLO
    │   │       ├── SSLYOLOv3
    │   │       ├── SSLYOLOv5
    │   │       ├── YOLOv3
    │   │       └── YOLOv5
    │   ├── README
    │   ├── REC
    │   │   └── CRNN
    │   │       └── CRNNv0
    │   └── RES
    │       └── MPRNet
    │           └── MPRNetv0
    ├── README.md
    └── WORKFLOW
        ├── CLS
        │   ├── TableCls
        │   │   └── v0
        │   └── TextCls
        │       └── v0
        ├── DET
        │   ├── SealDet
        │   │   ├── v0
        │   │   ├── v1
        │   │   └── v2
        │   ├── SealTextDet
        │   │   └── v0
        │   ├── TableCellDet
        │   │   ├── v0
        │   │   └── v1
        │   ├── TableDet
        │   │   ├── v0
        │   │   └── v1
        │   └── TextDet
        │       └── v0
        ├── OTHER
        │   ├── DeSealProcess
        │   │   └── v0
        │   ├── OCR
        │   │   ├── readme.txt
        │   │   ├── v0
        │   │   ├── v1
        │   │   └── v2
        │   └── TableStruct
        │       ├── v0
        │       ├── v1
        │       └── v2
        ├── REC
        │   └── TextRec
        │       └── v0
        ├── RES
        │   └── DeSeal
        │       └── v0
        └── SEG

