import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import datetime


class Log:
    def __init__(self, logger=None, output_path="/Logs"):
        filename = os.path.join(
            output_path, "log.txt"
        )  # '- + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '
        # 创建logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 定义输出格式
        format = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )

        # 创建输出到控制台handler sh
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(format)

        # 创建写入文件handler fh
        # fh = logging.FileHandler(filename=filename, encoding='utf-8', mode='w')
        fh = RotatingFileHandler(
            filename=filename,
            maxBytes=10 * 1024 * 1024,
            backupCount=1,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(format)

        # 给logger添加两个handler
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
