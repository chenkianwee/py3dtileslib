# -*- coding: utf-8 -*-
import struct
import numpy as np

from .tile import TileContent, TileHeader, TileBody, TileType
from .feature_table import FeatureTable
from .batch_table import BatchTable


class Pnts(TileContent):

    @staticmethod
    def from_features(features, pdtype = None, cdtype = None, ndtype = None, batchid_dtype = None, bt = None):
        """
        Parameters
        ----------
        features : Feature[]
        
        dtype : numpy.dtype
            Numpy description of a single feature
        
        cdtype : numpy.dtype
            Numpy description of a color
            
        ndtype : numpy.dtype
            Numpy description of a normal
        
        biddtype : numpy.dtype
            Numpy description of a batch id 
        
        bt : BatchTable
            Batchtable object 
            
        Returns
        -------
        tile : TileContent
        
        pdtype = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
        cdtype = np.dtype([('Red', np.uint8), ('Green', np.uint8), ('Blue', np.uint8)])
        position = np.array([(0,1,0)], dtype=dt)
        colour = np.array([(255,0,0)], dtype=dt2)
        
        # create a new feature from a uint8 numpy array
        pnt = py3dtileslib.Pnts.from_features(feature_list, pdtype = pdtype, cdtype = cdtype)
        
        """

        ft = FeatureTable.from_features(features, 'pnts', pdtype = pdtype, cdtype = cdtype, normal_dtype = ndtype, batchid_dtype = batchid_dtype)

        tb = PntsBody()
        tb.feature_table = ft
        tb.batch_table = bt

        th = PntsHeader()
        th.sync(tb)

        t = TileContent()
        t.body = tb
        t.header = th

        return t

    @staticmethod
    def from_array(array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        t : TileContent
        """

        # build tile header
        h_arr = array[0:PntsHeader.BYTELENGTH]
        h = PntsHeader.from_array(h_arr)

        if h.tile_byte_length != len(array):
            raise RuntimeError("Invalid byte length in header")

        # build tile body
        b_len = h.ft_json_byte_length + h.ft_bin_byte_length
        b_arr = array[PntsHeader.BYTELENGTH:PntsHeader.BYTELENGTH + b_len]
        b = PntsBody.from_array(h, b_arr)

        # build TileContent with header and body
        t = TileContent()
        t.header = h
        t.body = b

        return t

class PntsHeader(TileHeader):
    BYTELENGTH = 28

    def __init__(self):
        self.type = TileType.POINTCLOUD
        self.magic_value = b"pnts"
        self.version = 1
        self.tile_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0

    def to_array(self):
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array([self.version,
                                self.tile_byte_length,
                                self.ft_json_byte_length,
                                self.ft_bin_byte_length,
                                self.bt_json_byte_length,
                                self.bt_bin_byte_length], dtype=np.uint32)

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

    def sync(self, body):
        """
        Allow to synchronize headers with contents.
        """

        # extract array
        fth_arr = body.feature_table.header.to_array('pnts')
        ftb_arr = body.feature_table.body.to_array('pnts', len(fth_arr))

        # sync the tile header with feature table contents
        self.tile_byte_length = (len(fth_arr) + len(ftb_arr)
                                 + PntsHeader.BYTELENGTH)
        
        self.ft_json_byte_length = len(fth_arr)
        self.ft_bin_byte_length = len(ftb_arr)
        
        if body.batch_table is not None:
            ft_len = len(fth_arr) + len(ftb_arr)
            bth_arr = body.batch_table.header.to_array('pnts', ft_len)
            btb_arr = body.batch_table.body.to_array(body.batch_table.header, 'pnts', ft_len, len(bth_arr))
    
            self.tile_byte_length += len(bth_arr)
            self.bt_json_byte_length = len(bth_arr)
            
            self.tile_byte_length += len(btb_arr)
            self.bt_bin_byte_length = len(btb_arr)
            

    @staticmethod
    def from_array(array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        h : TileHeader
        """

        h = PntsHeader()

        if len(array) != PntsHeader.BYTELENGTH:
            raise RuntimeError("Invalid header length")

        h.magic_value = "pnts"
        h.version = struct.unpack("i", array[4:8])[0]
        h.tile_byte_length = struct.unpack("i", array[8:12])[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16])[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20])[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24])[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28])[0]

        h.type = TileType.POINTCLOUD

        return h

class PntsBody(TileBody):
    def __init__(self):
        self.feature_table = FeatureTable()
        self.batch_table = BatchTable()

    def to_array(self):
        ft_arr = self.feature_table.to_array('pnts')
        if self.batch_table is not None:
            ft_arr = np.concatenate((ft_arr, self.batch_table.to_array('pnts', len(ft_arr))))
            
        return ft_arr

    @staticmethod
    def from_array(th, array):
        """
        Parameters
        ----------
        th : TileHeader

        array : numpy.array

        Returns
        -------
        b : TileBody
        """
        # build feature table
        ft_len = th.ft_json_byte_length + th.ft_bin_byte_length
        ft_arr = array[0:ft_len]
        ft = FeatureTable.from_array(th, ft_arr)
        
        # build batch table       
        bt_len = th.bt_json_byte_length + th.bt_bin_byte_length
        bt_arr = array[ft_len:ft_len + bt_len]
        bt = BatchTable.from_array(th, ft, bt_arr)
        
        # build tile body with feature table
        b = PntsBody()
        b.feature_table = ft
        b.batch_table = bt

        return b
