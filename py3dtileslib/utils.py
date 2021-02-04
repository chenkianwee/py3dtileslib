# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
from pyproj import CRS, Transformer
from pygltflib import GLTF2

from .pnts import Pnts
from .b3dm import B3dm


class SrsInMissingException(Exception):
    pass


def convert_to_ecef(x, y, z, epsg_input):
    inp = CRS('epsg:{0}'.format(epsg_input))
    outp = CRS('epsg:4978')  # ECEF
    transformer = Transformer.from_crs(inp, outp)
    return transformer.transform(x, y, z)


class TileContentReader(object):

    @staticmethod
    def read_file(filename):
        with open(filename, 'rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return TileContentReader.read_array(arr)
        return None

    @staticmethod
    def read_array(array):
        magic = ''.join([c.decode('UTF-8') for c in array[0:4].view('c')])
        if magic == 'pnts':
            return Pnts.from_array(array)
        if magic == 'b3dm':
            return B3dm.from_array(array)
        return None
    
def glb2arr(gltf):
    """
    Convert GLTF2 object from pygltflib to numpy array
    
    Parameters
    ----------
    gltf : pygltflib.GLTF2

    Returns
    -------
    arr : numpy.array
    """
        
    # extract array
    #write to a temp file
    tmp_glb = tempfile.NamedTemporaryFile(suffix='.glb', prefix='py3dtiles_tempglb_', delete = False)
    tmp_glb_path = tmp_glb.name
    tmp_glb.close()
    gltf.save_binary(tmp_glb_path)
    
    with open(tmp_glb_path, 'rb') as f:
        data = f.read()
        glTF_arr = np.frombuffer(data, dtype=np.uint8)
    
    os.unlink(tmp_glb_path)
    return glTF_arr

def arr2gltf(gltf_arr):
    """
    Convert numpy array to GLTF2 object
    
    Parameters
    ----------
    arr : numpy.array

    Returns
    -------
    gltf : pygltflib.GLTF2
    
    """
    #write to a temp file
    tmp_glb = tempfile.NamedTemporaryFile(suffix='.glb', prefix='py3dtiles_tempglb_', delete = False)
    tmp_glb.write(bytes(gltf_arr))
    tmp_glb_path = tmp_glb.name
    tmp_glb.close()
    glTF = GLTF2().load(tmp_glb_path)
    os.unlink(tmp_glb_path)
    
    return glTF