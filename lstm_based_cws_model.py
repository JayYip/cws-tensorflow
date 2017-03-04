# -*- coding: utf-8 -*-

#Author: Jay Yip
#Date 04Mar2017

"""Chinese words segmentation model based on aclweb.org/anthology/D15-1141"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LSTMCWS(object):
    """docstring for LSTMCWS"""
    def __init__(self, arg):
        super(LSTMCWS, self).__init__()
        self.arg = arg
