from enum import Enum


class OCRType(str, Enum):
    AUTO = "auto"
    YES = "yes"
    NO = "no"


class Resolution(str, Enum):
    HIGH = "high"
    STD = "std"


class BorderTableAlg(str, Enum):
    SDBORDERTABLE_CV_MODEL = "sdbordertable_cv_model"


class BorderlessTableAlg(str, Enum):
    SDBORDERLESSTABLE_CV_LITE_MODEL = "sdborderlesstable_cv_lite_model"
    SDBORDERLESSTABLE_CV_MODEL = "sdborderlesstable_cv_model"
    SDBORDERLESSTABLE_LM_LITE_MODEL = "sdborderlesstable_lm_lite_model"
    SDBORDERLESSTABLE_LM_MODEL = "sdborderlesstable_lm_model"


class LayoutAlg(str, Enum):
    SDLAYOUT = "sdlayout"
    LAYOUTLMV3 = "layoutlmv3"


class ChapterAlg(str, Enum):
    RE = "re"  # 纯规则
    LAYOUT = "layout"  # 用layout的结果
    LLM = "llm"  # large language model
