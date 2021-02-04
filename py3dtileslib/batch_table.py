# -*- coding: utf-8 -*-

import numpy as np
import json


class BatchTable(object):
    """
    Only the JSON header has been implemented for now. According to the batch
    table documentation, the binary body is useful for storing long arrays of
    data (better performances)
    """

    def __init__(self):
        self.header = {}

    def add_property_from_array(self, propertyName, array):
        self.header[propertyName] = array

    # returns batch table as binary
    def to_array(self):
        # convert dict to json string
        bt_json = json.dumps(self.header, separators=(',', ':'))
        # header must be 4-byte aligned (refer to batch table documentation)
        bt_json += ' ' * (4 - len(bt_json) % 4)
        # returns an array of binaries representing the batch table
        return np.fromstring(bt_json, dtype=np.uint8)
    
    def from_array(self, array):
        """
        Parameters
        ----------
        array : numpy.array

        """
        bt_json_str = ''.join([c.decode('UTF-8') for c in array.view('c')])
        bt_json = json.loads(bt_json_str)
        self.header = bt_json