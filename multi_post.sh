#!/bin/bash

# 定义基本URL和输入文件夹
base_url="http://127.0.0.1:6000/upload_ocr_result"
input_folder="pytest_scripts/data/"

while true; do
  # 遍历文件夹中的所有文件
  for file in "$input_folder"/*; do
    # 获取文件名（不含路径）
    filename=$(basename "$file")

    # 构造输出文件名
    output_file="out.json"

    # 发送请求的函数
    send_request() {
      curl -X POST "$base_url" \
        -F "file=@$file" \
        -F "json={\"force_cv\": \"yes\", \"table_enhanced\": true}" \
        -o "$output_file"
    }

    # 启动并发请求
    send_request &
  done

  # 等待所有后台任务完成
  wait
  echo "所有请求完成"
  break
done