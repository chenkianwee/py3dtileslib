from typing import Union

import geomie3d
import py3dtileslib
import numpy as np
from pygltflib import GLTF2, Primitive, Buffer, BufferView, Accessor, ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER, UNSIGNED_SHORT, SCALAR, FLOAT

from . import utils

def add_cs_prim_outline(gltf: GLTF2) -> Union[int, None]:
    """
    takes in a gltf object and add the Cesium_primitive_outline extension into the gltf object

    Parameters
    ----------
    gltf: GLTF2
        gltf object to use.
 
    Returns
    -------
    buffer_indx : Union[int, None]
        the index of the buffer to use for storing the outline information.

    """
    buffer_indx = None
    ext_used = gltf.extensionsUsed
    if 'CESIUM_primitive_outline' not in ext_used:
        ext_used.append('CESIUM_primitive_outline')
        # add a new buffer to store the outline information
        buffer = Buffer()
        buffer_data = bytearray()
        py3dtileslib.utils.write_buffer(buffer, buffer_data)
        gltf.buffers.append(buffer)
        buffer_indx = len(gltf.buffers) - 1
    gltf.extensionsUsed = ext_used
    return buffer_indx

def add_outline2prim(gltf_prim: Primitive, gltf: GLTF2, buffer_indx: int):
    """
    takes in a gltf object and add the Cesium_primitive_outline extension into the primitives object
    
    Parameters
    ----------
    featureid : List[Union[float, int]]
        list of the feature id. Number of feature id needs to be the same as the number of vertices.
    """
    mode = gltf_prim.mode
    if mode != 4:
        raise ValueError('This is not a triangular mesh')
    
    indxs_orig = utils.get_idx_frm_primitive(gltf_prim, gltf)
    #===============================================================
    # find all the overlapping vertices and reindex them this is not necessary
    #===============================================================
    # uniq_pos = np.unique(pos, axis=0,return_inverse=True)
    # uniq1 = uniq_pos[1]
    # indxs_uniq = np.take(uniq1, indxs_orig)
    # #reindex the vertices
    # indxs = []
    # for iu in indxs_uniq:
    #     back_indxs = np.where(uniq1 == iu)[0]
    #     back_indx = back_indxs[0]
    #     indxs.append(back_indx)
    # indxs = np.reshape(indxs, (int(len(indxs)/3), 3))
    #===============================================================

    indxs_orig = np.reshape(indxs_orig, (int(len(indxs_orig)/3), 3))
    indxs = indxs_orig
    nrmls = utils.get_nmrl_frm_primitive(gltf_prim, gltf)
    tri_nrmls = np.take(nrmls, indxs_orig, axis=0)
    tri_nrmls2 = tri_nrmls[:,0]

    #grp the triangle surfaces according to normals
    uniq_tri_nrml = np.unique(tri_nrmls2, axis=0, return_inverse = True)
    nrml_idx = geomie3d.utility.separate_dup_non_dup(uniq_tri_nrml[1])
    non_dup_nidx = nrml_idx[0]
    dup_nidx = nrml_idx[1]
    outline_idxs = []
    if len(non_dup_nidx) != 0:
        indv_tri = np.take(indxs, non_dup_nidx, axis=0)
        outlines_tri = np.repeat(indv_tri, 2, axis=1)
        outlines_tri = np.roll(outlines_tri, -1, axis=1)
        outlines_tri_shape = np.shape(outlines_tri)
        outlines_tri = np.reshape(outlines_tri, (outlines_tri_shape[0]*3, 2)) # remove the trianges to just edges
        outline_idxs.extend(outlines_tri)
    
    if len(dup_nidx) != 0:
        for grp in dup_nidx: # compare the edges for each normal
            dup_tris = np.take(indxs, grp, axis=0) # take all the triangles that have the same normals
            edges_tri = np.repeat(dup_tris, 2, axis=1) # transform the triangles from vertices to edges 
            edges_tri = np.roll(edges_tri, -1, axis=1) # transform the triangles from vertices to edges
            edges_tri_shape = np.shape(edges_tri)
            edges_tri = np.reshape(edges_tri, (edges_tri_shape[0], 3, 2)) # transform the triangles from vertices to edges
            edges_tri_sort = np.sort(edges_tri) # sort the indices for identifying duplicates
            edges_tri_sort = np.reshape(edges_tri_sort, (edges_tri_shape[0]*3, 2)) # remove the trianges to just edges
            edges_tri = np.reshape(edges_tri, (edges_tri_shape[0]*3, 2)) # remove the trianges to just edges
            uniq_edges_tri = np.unique(edges_tri_sort, axis=0, return_inverse=True)
            non_dup_idx, dup_idx = geomie3d.utility.separate_dup_non_dup(uniq_edges_tri[1])
            srf_outlines = np.take(edges_tri, np.array(non_dup_idx, dtype='int'), axis=0)
            outline_idxs.extend(srf_outlines)

    # print(outline_idxs)
    outline_idxs = np.array(outline_idxs)
    outline_idxs_sort = np.sort(outline_idxs)
    uniq_edges = np.unique(outline_idxs_sort, axis=0, return_index=True) # find all the non overlapping outlines
    uniq_edges = np.take(outline_idxs, uniq_edges[1], axis=0)

    # uniq_edges = np.array([[0, 1], [1, 2], [2, 3], [3,0], [20, 21], [21, 22], [22, 23], [23, 20], [4, 7], [5, 6], [12,15], [13, 14]])
    # pos = utils.get_pos_frm_primitive(gltf_prim, gltf)
    # print(pos)
    # vs = geomie3d.create.vertex_list(pos)
    # geomie3d.viz.viz([{'topo_list': vs, 'colour': 'red'}])
    # pos2 = np.take(pos, uniq_edges, axis=0)
    # es = []
    # for e in pos2:
    #     vs = geomie3d.create.vertex_list(e)
    #     e1 = geomie3d.create.pline_edge_frm_verts(vs)
    #     es.append(e1)
    # geomie3d.viz.viz([{'topo_list': es, 'colour': 'blue'}])

    uniq_edges = uniq_edges.flatten()
    # print(uniq_edges)
    # get the data from the gltf file
    buffer = gltf.buffers[buffer_indx]
    data_arr = gltf.get_data_from_buffer_uri(buffer.uri)
    data_arr = bytearray(data_arr)

    #------------------------------------------------------------------------
    # pack the outline indices and extend it onto the data array
    # packed_outline_idxs = utils.pack_att(uniq_edges, '<H')
    packed_outline_idxs = utils.pack_att(uniq_edges, '<h')
    offset = len(data_arr)
    length = len(packed_outline_idxs)
    data_arr.extend(packed_outline_idxs)
    utils.write_buffer(buffer, data_arr)
    #------------------------------------------------------------------------ 
    # setup buffer view to read the data
    bufferView1 = BufferView()
    bufferView1.buffer = buffer_indx
    bufferView1.byteOffset = offset
    bufferView1.byteLength = length
    gltf.bufferViews.append(bufferView1)
    #------------------------------------------------------------------------ 
    # setup accessor to read the bufferview
    accessor1 = Accessor()
    accessor1.bufferView = len(gltf.bufferViews) - 1
    accessor1.byteOffset = 0
    accessor1.componentType = 5122 #5125 #UNSIGNED_SHORT 
    accessor1.count = len(uniq_edges)
    accessor1.type = SCALAR
    accessor1.max = [int(max(uniq_edges))]
    accessor1.min = [int(min(uniq_edges))]
    gltf.accessors.append(accessor1)
    #------------------------------------------------------------------------ 
    # add the attributes and extensions to the primitive
    exts = gltf_prim.extensions
    if 'CESIUM_primitive_outline' not in exts.keys():
        exts['CESIUM_primitive_outline'] = {'indices': len(gltf.accessors) - 1}
    else:
        print('Overwriting existing CESIUM_primitive_outline')
        exts['CESIUM_primitive_outline'] = {'indices': len(gltf.accessors) - 1}