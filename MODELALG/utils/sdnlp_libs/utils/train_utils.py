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
import random
import numpy as np
import torch


def setup_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

