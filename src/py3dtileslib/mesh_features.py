from typing import Union

import numpy as np
from pygltflib import GLTF2, Primitive, BufferView, Accessor, ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER, UNSIGNED_SHORT, SCALAR, FLOAT

from . import utils

def add_extmeshfeatures(gltf: GLTF2):
    """
    takes in a gltf object and add the EXT_mesh_features extension into the gltf object
    """
    ext_used = gltf.extensionsUsed
    if 'EXT_mesh_features' not in ext_used:
        ext_used.append('EXT_mesh_features')
    gltf.extensionsUsed = ext_used

def add_extmeshfeatures_by_vertex(gltf_prim: Primitive, gltf: GLTF2, featureid: list[Union[float, int]]):
    """
    takes in a gltf object and add the EXT_mesh_features extension into the gltf object
    
    Parameters
    ----------
    featureid : List[Union[float, int]]
        list of the feature id. Number of feature id needs to be the same as the number of vertices.
    """
    # get the data from the gltf file
    accessor = gltf.accessors[gltf_prim.attributes.POSITION]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data_arr = gltf.get_data_from_buffer_uri(buffer.uri)
    data_arr = bytearray(data_arr)
    #------------------------------------------------------------------------
    # pack the feature id and extend it onto the data array
    featureid = list(map(float, featureid))
    packed_feature_id = utils.pack_att(featureid, '<f')
    offset = len(data_arr)
    length = len(packed_feature_id)
    data_arr.extend(packed_feature_id)
    utils.write_buffer(buffer, data_arr)
    #------------------------------------------------------------------------ 
    # setup buffer view to read the data
    bufferView1 = BufferView()
    bufferView1.buffer = 0
    bufferView1.byteOffset = offset
    bufferView1.byteLength = length
    bufferView1.target = ARRAY_BUFFER # ELEMENT_ARRAY_BUFFER 
    gltf.bufferViews.append(bufferView1)
    #------------------------------------------------------------------------ 
    # setup accessor to read the bufferview
    accessor1 = Accessor()
    accessor1.bufferView = len(gltf.bufferViews) - 1
    accessor1.byteOffset = 0
    accessor1.componentType = FLOAT
    accessor1.count = len(featureid)
    accessor1.type = SCALAR
    accessor1.normalized = False
    accessor1.max = [int(max(featureid))]
    accessor1.min = [int(min(featureid))]
    gltf.accessors.append(accessor1)
    #------------------------------------------------------------------------ 
    # add the attributes and extensions to the primitive
    att = gltf_prim.attributes
    att._FEATURE_ID_0 = len(gltf.accessors) - 1
    uniq = np.unique(featureid)
    featureIds_json = {"featureIds": [{"featureCount": len(uniq), "attribute": 0 }]}
    exts = gltf_prim.extensions
    if 'EXT_mesh_features' not in exts.keys():
        exts['EXT_mesh_features'] = featureIds_json
    else:
        print('Overwriting existing EXT mesh features')
        exts['EXT_mesh_features'] = featureIds_json
    