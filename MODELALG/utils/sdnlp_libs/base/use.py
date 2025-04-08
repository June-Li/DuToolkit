#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#
# Copyright (C) 2022 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#
# Author: Li Xiuming
# Last Modified: 2022-03-15
############################################################

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class USEInfer:
    def __init__(self, tfhub_cache_dir=None):
        # 加载use模型
        if tfhub_cache_dir is None:
            os.environ["TFHUB_CACHE_DIR"] = "/home/lixm/common_libraries/pretrained_model/tfhub_cache"
        else:
            os.environ["TFHUB_CACHE_DIR"] = tfhub_cache_dir
        module_url = "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-multilingual/3"
        self.embed = hub.load(module_url)

    def apply(self, sentences):
        embeds = self.embed(sentences)
        embeds = embeds.numpy()
        return embeds

    def get_topn_sim(self, sentence_to_compare, sentences, topn=None):
        embedding_to_compare = self.embed([sentence_to_compare])
        embeddings = self.embed(sentences)

        sim_matrix = cosine_similarity(embedding_to_compare, embeddings)

        topn_sents = []
        if topn is None:
            topn = len(sentences)
        topn = min(len(sentences), topn)
        topn_idx = np.argpartition(sim_matrix[0], -topn)[-topn:]

        for idx in topn_idx:
            topn_sents.append((str(sim_matrix[0][idx]), idx, sentences[idx]))

        topn_sents.sort(key=lambda x: np.float(x[0]), reverse=True)

        return topn_sents

if __name__ == "__main__":
    sentences = ["20亿商业票据逾期，荣盛“以房抵债”",
                 "广汽集团公告，1-4月累计产量为63.9万辆，同比下降4.9%；累计销量为65.1万辆，同比下降3.17%。', '广汽集团】【广汽集团前4个月汽车销量同比下降4.9%"]
    use_inferer = USEInfer()
    embeds = use_inferer.get_sent_embed(sentences)
    print(embeds)

