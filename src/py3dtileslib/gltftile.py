import struct
import numpy as np

from . import utils 
from .b3dm import B3dmBody, B3dmHeader
from .tile import TileContent

class GltfTile(object):

    @staticmethod
    def from_glTF(gltf, bt=None, ft=None):
        """
        Parameters
        ----------
        gltf : GlTF2
            glTF2 object from the pygltflib

        bt : Batch Table (optional)
            BatchTable object containing per-feature metadata
            
        ft : Feature Table (optional)
            FeatureTable object containing per-feature metadata
            
        Returns
        -------
        tile : TileContent
        """

        tb = B3dmBody()
        tb.glTF = gltf
        tb.batch_table = bt
        tb.feature_table = ft

        th = B3dmHeader()
        th.sync(tb)

        t = TileContent()
        t.body = tb
        t.header = th

        return t