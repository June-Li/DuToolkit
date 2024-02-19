""" dynamically load settings

author baiyu
"""
import WORKFLOW.CLS.TableCls.v0.config.global_settings as settings


class Settings:
    def __init__(self, settings):
        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))


settings = Settings(settings)
