# coding: utf-8
import struct
import numpy as np

from .tile import TileContent, TileHeader, TileBody, TileType
from .batch_table import BatchTable
from .feature_table import FeatureTable
from . import utils 

class I3dm(TileContent):

    @staticmethod
    def from_glTF(gltf, bt=None, ft=None):
        """
        Parameters
        ----------
        gltf : GlTF2
            glTF2 object from the pygltflib

        bt : Batch Table (optional)
            BatchTable object containing per-feature metadata
            
        ft : Feature Table
            FeatureTable object containing per-feature metadata
            
        Returns
        -------
        tile : TileContent
        """

        tb = I3dmBody()
        tb.glTF = gltf
        tb.batch_table = bt
        tb.feature_table = ft

        th = I3dmHeader()
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
        h_arr = array[0:I3dmHeader.BYTELENGTH]
        h = I3dmHeader.from_array(h_arr)

        if h.tile_byte_length != len(array):
            raise RuntimeError("Invalid byte length in header")

        # build tile body
        b_arr = (array[I3dmHeader.BYTELENGTH:h.tile_byte_length])
        b = I3dmBody.from_array(h, b_arr)

        # build TileContent with header and body
        t = TileContent()
        t.header = h
        t.body = b

        return t


class I3dmHeader(TileHeader):
    BYTELENGTH = 32

    def __init__(self):
        self.type = TileType.INSTANCED3DMODEL
        self.magic_value = b"i3dm"
        self.version = 1
        self.tile_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0
        self.gltf_format = 1
        self.bt_length = 0  # number of models in the batch

    def to_array(self):
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array([self.version,
                                self.tile_byte_length,
                                self.ft_json_byte_length,
                                self.ft_bin_byte_length,
                                self.bt_json_byte_length,
                                self.bt_bin_byte_length,
                                self.gltf_format], dtype=np.uint32)

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

    def sync(self, body):
        """
        Allow to synchronize headers with contents.
        """

        # extract array
        glTF_arr = utils.glb2arr(body.glTF)

        # sync the tile header with feature table contents
        self.tile_byte_length = len(glTF_arr) + I3dmHeader.BYTELENGTH
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.gltf_format = 1
        
        if body.batch_table is not None:
            bth_arr = body.batch_table.to_array()
            # btb_arr = body.batch_table.body.to_array()

            self.tile_byte_length += len(bth_arr)
            self.bt_json_byte_length = len(bth_arr)
            
        if body.feature_table is not None:
            fth_arr = body.feature_table.header.to_array()
            ftb_arr = body.feature_table.body.to_array()
            self.tile_byte_length += len(fth_arr)
            self.ft_json_byte_length = len(fth_arr)
            
            self.tile_byte_length += len(ftb_arr)
            self.ft_bin_byte_length = len(ftb_arr)

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

        h = I3dmHeader()

        if len(array) != I3dmHeader.BYTELENGTH:
            raise RuntimeError("Invalid header length")

        h.magic_value = "i3dm"
        h.version = struct.unpack("i", array[4:8])[0]
        h.tile_byte_length = struct.unpack("i", array[8:12])[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16])[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20])[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24])[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28])[0]
        h.gltf_format = struct.unpack("i", array[28:32])[0]
        h.type = TileType.INSTANCED3DMODEL

        return h

class I3dmBody(TileBody):
    def __init__(self):
        self.batch_table = BatchTable()
        self.feature_table = FeatureTable()
        self.glTF = None

    def to_array(self):
        array = utils.glb2arr(self.glTF)
        if self.batch_table is not None:
            array = np.concatenate((self.batch_table.to_array(), array))
            
        if self.feature_table is not None:
            array = np.concatenate((self.feature_table.to_array(), array))
            
        return array

    @staticmethod
    def from_glTF(glTF):
        """
        Parameters
        ----------
        th : TileHeader

        glTF : GlTF2

        Returns
        -------
        b : TileBody
        """

        # build tile body
        b = I3dmBody()
        b.glTF = glTF

        return b

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
        bt_json_len = th.bt_json_byte_length
        bt_json_arr = array[ft_len:ft_len + bt_json_len]
        
        # bt_bin_len = th.bt_bin_byte_length
        # bt_bin_arr = array[ft_len + bt_json_len:ft_len + bt_json_len + bt_bin_len]
        
        bt = BatchTable()
        bt.from_array(bt_json_arr)

        # build glTF
        bt_len = th.bt_json_byte_length + th.bt_bin_byte_length
        glTF_len = (th.tile_byte_length - ft_len - bt_len
                    - I3dmHeader.BYTELENGTH)
        glTF_arr = array[ft_len + bt_len:ft_len + bt_len + glTF_len]
        glTF = utils.arr2gltf(glTF_arr)
    
        # build tile body with feature table
        b = I3dmBody()
        b.feature_table = ft
        b.batch_table = bt
        b.glTF = glTF

        return b