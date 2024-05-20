# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
-------------------------------------------------
   Description :
   Author :       caiyueliang
   Date :         2023/02/26
-------------------------------------------------

"""
import time
import threading
# from loguru import logger
import logging as logger
# from taichu.utils import log_utils as logger


class TimeUtils(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if self._init_flag is False:
            logger.info('[TimeUtils] init start ...')
            # self.name = name
            self.task_dict = dict()
            # self.cost_list = []
            self._init_flag = True
            logger.info('[TimeUtils] init end ...')

    def __new__(cls, *args, **kwargs):
        if not hasattr(TimeUtils, "_instance"):
            with TimeUtils._instance_lock:
                if not hasattr(TimeUtils, "_instance"):
                    TimeUtils._instance = object.__new__(cls)
        return TimeUtils._instance

    def start(self, task_name: str = "default"):
        # self.cost_list = []
        # self.cost_list.append(("start", time.time()))
        self.task_dict[task_name] = list()
        self.task_dict[task_name].append(("start", time.time()))

    def append(self, step_name: str, task_name: str = "default"):
        if task_name in self.task_dict.keys():
            # self.cost_list.append((step_name, time.time()))
            self.task_dict[task_name].append((step_name, time.time()))
        else:
            logger.warning("[append] task_name: {} not exist. ".format(task_name))

    def print(self, task_name: str = "default"):
        if task_name in self.task_dict.keys():
            time_str = "[{}]".format(task_name)
            for i in range(len(self.task_dict[task_name]) - 1):
                time_str += "[{}] {}s; ".format(
                    self.task_dict[task_name][i+1][0], round(self.task_dict[task_name][i+1][1] - self.task_dict[task_name][i][1], 5))
            logger.warning(time_str)
            logger.warning("[{}][total] {}s".format(task_name, round(self.task_dict[task_name][-1][1] - self.task_dict[task_name][0][1], 5)))
            self.task_dict.pop(task_name)
        else:
            logger.warning("[print] task_name: {} not exist. ".format(task_name))
