# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 14:29
# @Author  : zhoujun
import time

import numpy as np
import torch


class CTCLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = []
        with open(character, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                dict_character += list(line)
        # dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        # TODO replace ‘ ’ with special symbol
        self.character = (
            ["[blank]"] + dict_character + [" "]
        )  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=None):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        # text = ''.join(text)
        # text = [self.dict[char] for char in text]
        d = []
        batch_max_length = max(length)
        for s in text:
            t = [self.dict[char] for char in s]
            t.extend([0] * (batch_max_length - len(s)))
            d.append(t)
        return (
            torch.tensor(d, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

    def decode(self, preds, raw=False):
        """convert text-index into text-label."""
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            if raw:
                result_list.append(
                    ("".join([self.character[int(i)] for i in word]), prob)
                )
            else:
                result = []
                conf = []
                for i, index in enumerate(word):
                    if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                        result.append(self.character[int(index)])
                        conf.append(prob[i])
                result_list.append(("".join(result), conf))
        return result_list

    def decode_by_tensor(self, preds, raw=False):
        """convert text-index into text-label."""
        # totaltime = time.time()
        # s = time.time()
        preds_info = preds.max(axis=2)
        # print("max: ", time.time() - s)
        result_list = []
        # s = time.time()
        indices, values = (
            preds_info.indices.detach().cpu().numpy(),
            preds_info.values.detach().cpu().numpy(),
        )
        # print("to cpu: ", time.time() - s)
        for idx in range(preds_info.values.shape[0]):
            word, prob = indices[idx], values[idx]
            if raw:
                result_list.append(("".join([self.character[i] for i in word]), prob))
            else:
                # s = time.time()
                result, conf = [], []
                # nonzero_indices = word.nonzero()
                nonzero_indices = np.where(word > 0)[0]
                # print("nonzero: ", time.time() - s)
                # s = time.time()
                for idx_1 in nonzero_indices:
                    if not (idx_1 > 0 and word[idx_1 - 1] == word[idx_1]):
                        result.append(self.character[word[idx_1]])
                        conf.append(float(prob[idx_1]))
                result_list.append(("".join(result), conf))
                # print("search: ", time.time() - s)
        # print("结果处理总耗时: ", time.time() - totaltime)
        return result_list
