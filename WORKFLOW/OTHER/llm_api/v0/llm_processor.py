import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import json
import logging
import random
import time
from typing import Literal

import requests
from diskcache import Cache
from dynaconf import Dynaconf

from MODELALG.utils import common

# 请替换为你自己的KEY
API_KEY = os.getenv("OPENAI_API_KEY", "")

rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(rootdir)


def get_message(
    task_description, question, question_type: Literal["base64_image", "text"]
):
    if question_type == "base64_image":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_description},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{question}"},
                    },
                ],
            }
        ]
    elif question_type == "text":
        return [
            {
                "role": "system",
                "content": task_description,  # "你是ChatGPT, 一个由OpenAI训练的大型语言模型, 你旨在回答并解决人们的任何问题，并且可以使用多种语言与人交流。",
            },
            {"role": "user", "content": f"{question}"},
        ]
    else:
        raise ValueError(f"Invalid question type: {question_type}")


def get_answer(
    question,
    question_type: Literal["base64_image", "text"],
    task_description="你是ChatGPT, 一个由OpenAI训练的大型语言模型, 你旨在回答并解决人们的任何问题，并且可以使用多种语言与人交流。",
    model="qwen2-72b-chat",
    url="https://gateway.chat.sensedeal.vip/v1/chat/completions",
    max_retries=2,
):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "messages": get_message(task_description, question, question_type),
        "temperature": 0.01,
        "top_p": 1,
    }
    data = json.dumps(data)

    # 加入x次重试
    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            if i == max_retries - 1:
                raise e
            else:
                # 每次等待时间
                wait_time = (2**i) + random.random()
                logging.info(f"Retry #{i + 1} after {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    ret = response.json()["choices"][0]["message"]["content"]
    # logging.debug(f"[api/chatgpt]回答: {ret}")
    return ret


if __name__ == "__main__":
    from datetime import datetime

    import cv2

    from MODELALG.utils.common import encode_image

    # 获取配置
    default_settings = Dynaconf(settings_files=[root_dir + "/system.yml"]).to_dict()
    if os.environ.get("ENV_FOR_OCR") == "development":
        configs = default_settings
    else:
        production_settings = Dynaconf(
            settings_files=[root_dir + "/system.production.yml"]
        ).to_dict()
        configs = common.dict_deep_update(default_settings, production_settings)

    # 获取问题
    if configs["PROCESSES_CONTROL"]["Figure"]["vl"]["use_flag"]:
        task_description = "请详细描述一下图片中的内容："
        base64_image = common.encode_image(
            cv2.imread("/volume/test_data/fake/table_1.jpg")
        )
        question_type = "base64_image"
        try:
            A = get_answer(
                base64_image,
                question_type,
                task_description,
                model=configs["PROCESSES_CONTROL"]["Figure"]["vl"]["model"],
                url=configs["API_BASE"],
                max_retries=configs["PROCESSES_CONTROL"]["Figure"]["vl"]["max_retries"],
            )
            print("Q: ", task_description)
            print("A: ", A)
            print("chatgpt调用成功！！！")
            print(type(A))
        except:
            print("chatgpt不可调用")
            configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["mul_line_merge"] = False

    if configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["mul_line_merge"]:
        try:
            Q = "hi"
            A = get_answer(
                question=Q,
                question_type="text",
                task_description="你是ChatGPT, 一个由OpenAI训练的大型语言模型, 你旨在回答并解决人们的任何问题，并且可以使用多种语言与人交流。",
                model=configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["model"],
                url=configs["API_BASE"],
                max_retries=configs["PROCESSES_CONTROL"]["TableModule"]["GPT"][
                    "max_retries"
                ],
            )
            print("Q: ", Q)
            print("A: ", A)
            print("chatgpt调用成功！！！")
        except:
            print("chatgpt不可调用")
            configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["mul_line_merge"] = False
