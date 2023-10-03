# -*- coding: utf-8 -*-
import os
import math
import base64
import struct
import tempfile
from typing import Union

from scipy.spatial.transform import Rotation
import numpy as np
import geomie3d

from pyproj import Transformer
from pygltflib import GLTF2, Primitive, Buffer, Mesh, Node

from .pnts import Pnts
from .b3dm import B3dm
from .i3dm import I3dm

def get_pos_frm_primitive(gltf_prim: Primitive, gltf: GLTF2) -> np.ndarray:
    """
    get all the vertices from the gltf.primitive
    
    Returns
    -------
    verts : np.ndarray(Shape[Any,3])
    the vertices of the primitive
    """
    verts = []
    accessor = gltf.accessors[gltf_prim.attributes.POSITION]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    # pull each vertex from the binary buffer and convert it into a tuple of python floats
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*12  # the location in the buffer of this vertex
        d = data[index:index+12]  # the vertex data
        v = struct.unpack("<fff", d)   # convert from base64 to three floats
        verts.append(v)
    
    return np.array(verts)

def get_nmrl_frm_primitive(gltf_prim: Primitive, gltf: GLTF2) -> np.ndarray:
    """
    get all the normals from the gltf.primitive
    
    Returns
    -------
    nrmls : np.ndarray(Shape[Any,3])
    the nrmls of the vertices of the primitive
    """
    nrmls = []
    accessor = gltf.accessors[gltf_prim.attributes.NORMAL]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    # pull each vertex from the binary buffer and convert it into a tuple of python floats
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*12  # the location in the buffer of this vertex
        d = data[index:index+12]  # the nrml data
        v = struct.unpack("<fff", d)   # convert from base64 to three floats
        nrmls.append(v)
    
    return np.array(nrmls)

def get_idx_frm_primitive(gltf_prim: Primitive, gltf: GLTF2) -> np.ndarray:
    """
    get all the indices from the gltf.primitive
    
    Returns
    -------
    idx : np.ndarray
    the indices of the primitive
    """
    indxs = []
    accessor = gltf.accessors[gltf_prim.indices]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    # pull each vertex from the binary buffer and convert it into a tuple of python floats
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*2  # the location in the buffer of this vertex
        d = data[index:index+2]  # the index data
        idx = struct.unpack("<H", d)   # convert from base64 to int
        indxs.extend(idx)
    
    return np.array(indxs)

def get_pos_frm_mesh(gltf_mesh: Mesh, gltf: GLTF2) -> np.ndarray:
    """
    get all the vertices from the gltf mesh

    Returns
    -------
    verts : np.ndarray(Shape[Any,3])
    the vertices of the mesh
    """
    prims = gltf_mesh.primitives
    vert_ls = []
    for prim in prims:
        verts = get_pos_frm_primitive(prim, gltf)
        vert_ls.extend(verts)
    return np.array(vert_ls)

def get_pos_from_node(gltf_node: Node, gltf: GLTF2) -> np.ndarray:
    """
    get all the vertices from the gltf node

    Returns
    -------
    verts : np.ndarray(Shape[Any,3])
    the vertices of the node
    """
    meshes = gltf.meshes
    nodes = gltf.nodes
    mesh_id = gltf_node.mesh
    children_node_ids = gltf_node.children

    # the transformation order is always scale, rotate and translate
    node_mat = gltf_node.matrix
    if node_mat is not None:
        mat = np.array(node_mat)
        mat = np.reshape(mat, (4,4))
    else:
        mat = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        scale = gltf_node.scale
        rot = gltf_node.rotation
        trsl = gltf_node.translation
        if trsl is not None:
            trsl_mat = geomie3d.calculate.translate_matrice(trsl[0], trsl[1], trsl[2])
            mat = mat@trsl_mat

        if rot is not None:
            r = Rotation.from_quat(rot)
            rot_mat = r.as_matrix()
            rot_mat = np.array([[rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], 0],
                                [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], 0],
                                [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], 0],
                                [0,0,0,1]])
            mat = mat@rot_mat

        if scale is not None:
            scale_mat = geomie3d.calculate.scale_matrice(scale[0], scale[1], scale[2])
            mat = mat@scale_mat
    
    vert_ls = []
    if mesh_id is not None:
        sel_mesh = meshes[mesh_id]
        verts = get_pos_frm_mesh(sel_mesh, gltf)
        verts = geomie3d.calculate.trsf_xyzs(verts, mat)
        vert_ls.extend(verts)
    
    if len(children_node_ids) != 0:
        for cn_id in children_node_ids:
            sel_node = nodes[cn_id]
            verts2 = get_pos_from_node(sel_node, gltf)
            vert_ls.extend(verts2)
    return np.array(vert_ls)

def get_pos_frm_gltf(gltf: GLTF2) -> np.ndarray:
    """
    get all the vertices from the gltf

    Returns
    -------
    verts : np.ndarray(Shape[Any,3])
    the vertices of the gltf file
    """
    default_scene_id = gltf.scene
    scenes = gltf.scenes
    sel_scene = scenes[default_scene_id]
    scene_node_ids = sel_scene.nodes
    nodes = gltf.nodes
    vert_ls = []
    for sid in scene_node_ids:
        sel_node = nodes[sid]
        verts = get_pos_from_node(sel_node, gltf)
        vert_ls.extend(verts)
    return vert_ls

def define_bbox(midpt: list[Union[float, int]], xdim: Union[float, int], ydim: Union[float, int], 
                zdim: Union[float, int], xdir: list[Union[float, int]] = [1.0, 0.0, 0.0], ydir: list[Union[float, int]] = [0.0, 1.0, 0.0],
                zdir: list[Union[float, int]] = [0.0, 0.0, 1.0], geo_loc: list[Union[float, int]] = None) -> list[float]:
    """
    Defines the bbox for the 3dtiles specification.
    
    Parameters
    ----------
    midpt : list[Union[float, int]]
    the midpoint [x, y, z] of the bounding box. The [x, y, z] is a point in the gltf model.
    
    xdim :Union[float, int]
    the length of the bounding box in the x direction (meters).

    ydim :Union[float, int]
    the length of the bounding box in the y direction (meters).

    zdim :Union[float, int]
    the length of the bounding box in the z direction (meters).

    xdir :list[Union[float, int]]
    the x direction of the bounding box.

    ydir :list[Union[float, int]]
    the y direction of the bounding box.

    zdir :list[Union[float, int]]
    the z direction of the bounding box.

    geo_loc : list[Union[float, int]], optional
    the geo_loc [lon (deg), lat (deg), altitude (m)].The geo-ref position of the midpt.

    Returns
    -------
    bbox : list[float]
    list of 12 numbers that define the bbox for bvol of a 3dtile
    """
    bbox = []
    bbox.extend(midpt)
    xaxis = geomie3d.calculate.move_xyzs([0,0,0], [xdir], [xdim/2])[0]
    yaxis = geomie3d.calculate.move_xyzs([0,0,0], [ydir], [ydim/2])[0]
    zaxis = geomie3d.calculate.move_xyzs([0,0,0], [zdir], [zdim/2])[0]
    bbox.extend(xaxis)
    bbox.extend(yaxis)
    bbox.extend(zaxis)
    return bbox

def geodetic2ecef(geo_loc: list[Union[float, int]], from_epsg: str = 'EPSG:4326', to_epsg: str = 'EPSG:4978') -> list[float]:
    """
    Convert geodetic to ECEF.
    
    Parameters
    ----------
    geo_loc : list[Union[float, int]], optional
    the geo_loc [lon (deg), lat (deg), altitude (m)].

    from_epsg : str, optional
        the epsg string to project from. EPSG:4326 is the common crs used in google maps.
    
    to_epsg : str, optional
        the epsg string to project to. EPSG:4978 is a ecef/geocentric crs for the world https://epsg.io/4978

    Returns
    -------
    ecef_coord : list[float]
    the ecef coordinate.
    
    """
    # ecef = pyproj.crs.GeocentricCRS(datum=datum)
    # lla2ecef = Transformer.from_crs(epsg, ecef,always_xy=True)
    # ecef_coord = lla2ecef.transform(geo_loc[0], geo_loc[1], geo_loc[2])

    epsg4326_4978 = Transformer.from_crs(from_epsg, to_epsg, always_xy = True)
    ecef_coord = epsg4326_4978.transform(geo_loc[0], geo_loc[1], geo_loc[2])
    return list(ecef_coord)

def gcs2pcs(geo_loc: list[Union[float, int]], from_epsg: str = 'EPSG:4326', to_epsg: str = 'EPSG:4087') -> list[float]:
    """
    covert a geographic coordinate system in degrees to projected coordinate system in meters.

    Parameters
    ----------
    geo_loc : list[Union[float, int]], optional
        the geo_loc [lon (deg), lat (deg), altitude (m)].

    from_epsg : str
        the epsg string to project from. EPSG:4326 is the common crs used in google maps.
    
    to_epsg : str
        the epsg string to project to. EPSG:4978 is a projected crs for the world https://epsg.io/4087

    Returns
    -------
    geo_loc_xyz : list
        list of three elements of the projected coordinate in xyz meters
    """
    epsg4326_4978 = Transformer.from_crs(from_epsg, to_epsg, always_xy = True)
    geo_loc_xyz = epsg4326_4978.transform(geo_loc[0], geo_loc[1], geo_loc[2])
    return geo_loc_xyz

def pcs2gcs(geo_loc_xyz: list[Union[float, int]], from_epsg: str = 'EPSG:4087', to_epsg: str = 'EPSG:4326') -> list[float]:
    """
    covert a geographic coordinate system in degrees to projected coordinate system in meters.

    Parameters
    ----------
    geo_loc_xyz : list[Union[float, int]], optional
        the geo_loc [lon (m), lat (m), altitude (m)].

    from_epsg : str
        the epsg string to project from. EPSG:4978 is a projected crs for the world https://epsg.io/4087
    
    to_epsg : str
        the epsg string to project to. EPSG:4326 is the common crs used in google maps.

    Returns
    -------
    geo_loc : list
        list of three elements of the gcs.
    """
    epsgfrom_to = Transformer.from_crs(from_epsg, to_epsg, always_xy = False)
    geo_loc = epsgfrom_to.transform(geo_loc_xyz[0], geo_loc_xyz[1], geo_loc_xyz[2])
    if to_epsg == 'EPSG:4326':
        geo_loc = [geo_loc[1], geo_loc[0], geo_loc[2]]
    return geo_loc

def compute_trsfmat4enu_frm_gcs_coord(geo_loc: list[Union[float, int]], epsg: str = "EPSG:4326", datum: str = 'urn:ogc:def:datum:EPSG::6326') -> np.ndarray:
    """
    Compute the 4x4 transformation Matrix of transforming the model to East North Up (ENU) coordinate system with a geodetic coordinate origin.
    - https://gis.stackexchange.com/questions/308445/local-enu-point-of-interest-to-ecef
    - https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    
    Parameters
    ----------
    geo_loc : tuple
    Tuple of 3 specifying the Lon/Lat/Altitude in degrees and metres.

    epsg : str, optional
    the epsg of the CRS of the geo_loc.

    datum : str, optional
    the datum of the ecef crs.

    Returns
    -------
    trsf_mat_3dtiles : np.ndarray
    2 dimensional matrix of 4x4 transformation
    
    """
    lla_origin = geo_loc

    # epsg4326_4978 = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
    # ecef_origin = epsg4326_4978.transform(lla_origin[0], lla_origin[1], lla_origin[2])

    ecef_origin = geodetic2ecef(geo_loc)
    
    lon = math.radians(lla_origin[0])
    lat = math.radians(lla_origin[1])
    sin_lon = math.sin(lon)
    sin_lat = math.sin(lat)
    cos_lon = math.cos(lon)
    cos_lat = math.cos(lat)
    
    r1 = sin_lon*-1
    r2 = (sin_lat*-1)*cos_lon
    r3 = cos_lat*cos_lon
    r4 = cos_lon
    r5 = (sin_lat*-1)*sin_lon
    r6 = cos_lat*sin_lon
    r7 = 0
    r8 = cos_lat
    r9 = sin_lat
    
    tx = ecef_origin[0]
    ty = ecef_origin[1]
    tz = ecef_origin[2]
    
    trsf_mat = np.array([[r1,r2,r3,tx],
                        [r4,r5,r6,ty],
                        [r7,r8,r9,tz],
                        [0,0,0,1]])
    return trsf_mat

def trsf3dtiles2_np_mat(trsf3dtiles: np.ndarray) -> np.ndarray:
    """
    convert transformation matrix from 3dtiles format to a format that can be used in numpy

    Parameters
    ----------
    trsf3dtiles : np.ndarray
    3dtiles transfer matrix from the function compute_trsfmat4enu_frm_gcs_coord()

    Returns
    -------
    npmat : np.ndarray
    2 dimensional matrix of 4x4 transformation ready for the trsf_pos() function.
    
    """
    npmat = np.array([[trsf3dtiles[0][0], trsf3dtiles[0][1], trsf3dtiles[0][2], 0],
                      [trsf3dtiles[1][0], trsf3dtiles[1][1], trsf3dtiles[1][2], 0],
                      [trsf3dtiles[2][0], trsf3dtiles[2][1], trsf3dtiles[2][2], 0],
                      [trsf3dtiles[3][0], trsf3dtiles[3][1], trsf3dtiles[3][2], 1]])
    
    return npmat

def trsf_pos(pos_list: np.ndarray, trsf_mat: np.ndarray) -> np.ndarray:
    """
    Transform the positions according to the transformation matrix
    
    Parameters
    ----------
    pos_list : np.ndarray
    array of vertices of the gltf file must be z-up.
    
    trsf_mat : np.ndarray
    2 dimensional matrix of 4x4 transformation

    Returns
    -------
    trsf_pos_list : np.ndarray
    Transformed position list
    
    """
    if type(pos_list) != np.ndarray:
        np.array(pos_list)
    
    #add an extra column to the points
    npos = len(pos_list)
    xyzw = np.ones((npos,4))
    xyzw[:,:-1] = pos_list
    t_xyzw = np.dot(xyzw, trsf_mat.T)
    trsf_xyzs = t_xyzw[:,:-1]
    
    return trsf_xyzs.tolist()

def pack_att(att_list: list, pack_format: str) -> bytearray:
    """
    refer to https://docs.python.org/3/library/struct.html to understand the various pack format.

    Parameters
    ----------
    pack_format : str
        string specifying the type of byte to pack into. e.g. '<ffff' for pos_list and nrml_list, '<H' for indices, '<c' for strings
 
    Returns
    -------
    packed_list : bytearray
        the packed data
    """
    if type(att_list) == np.ndarray:
        att_list = att_list.tolist()
    pack_res = bytearray()
    for att in att_list:
        if type(att) == list or type(att) == tuple: 
            pack = struct.pack(pack_format, *att)
        else:
            pack = struct.pack(pack_format, att)
        pack_res.extend(pack)
    return pack_res

def pack_att_string(att_list: list[str]) -> tuple[bytearray, list[int]]:
    """
    refer to https://docs.python.org/3/library/struct.html to understand the various pack format.

    Parameters
    ----------
    att_list : list[str]
        list of string to be packed
 
    Returns
    -------
    packed_list : bytearray
        the packed data

    offset : list[int]
        the string offset as required by the propertytables
    """
    if type(att_list) == np.ndarray:
        att_list = att_list.tolist()
    pack_res = bytearray()
    offset = [0]
    char_cnt = 0
    for att in att_list:
        nchar = len(att)
        char_cnt+=nchar
        offset.append(char_cnt)
        pack_format = '<' + str(nchar) + 's'
        pack = struct.pack(pack_format, att.encode('utf-8'))
        pack_res.extend(pack)
    return pack_res, offset

def write_buffer(buffer: Buffer, data_arr: bytearray):
    buffer_data_header = 'data:application/octet-stream;base64,'
    data = base64.b64encode(data_arr).decode('utf-8') 
    buffer_data = f'{buffer_data_header}{data}'
    buffer.uri = buffer_data
    buffer.byteLength = len(data_arr)

def compute_error(bbox):
    """
    Compute the error of the tile based on its bbox
    
    Parameters
    ----------
    bbox : tuple
    Tuple of 6 numbers [mnx,mny,mnz,mxx,mxy,mxz] 

    Returns
    -------
    error : float
    The geometry error of the tile
    
    """
    lwr_left = [bbox[0], bbox[1], bbox[2]]
    lwr_right =  [bbox[3], bbox[4], bbox[2]]
    error = geomie3d.calculate.dist_btw_xyzs(lwr_left, lwr_right)
    return error

def get_outline_indx(gltf_prim: Primitive, gltf: GLTF2) -> np.ndarray:
    """
    get all the edge indices from the gltf.primitive cesium edge outline
    
    Returns
    -------
    idx : np.ndarray
    the indices of the edge outline
    """
    indxs = []
    accessor = gltf.accessors[gltf_prim.extensions['CESIUM_primitive_outline']['indices']]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    # pull each vertex from the binary buffer and convert it into a tuple of python floats
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*2  # the location in the buffer of this vertex
        d = data[index:index+2]  # the index data
        idx = struct.unpack("<h", d)   # convert from base64 to int
        indxs.extend(idx)
    
    return np.array(indxs)
#=====================================================================================================================
# region: Legacy code to be deprecated
#=====================================================================================================================
def compute_tile_bbox(pos_list, geo_loc = None):
    """
    This function only works if the gltf content is in z-up or y-up coordinate system, we are assuming the gltf has z-up coordinates while being transform to y-up in the gltf file
    in the root node of the gltf file. So we need to transform it to y-up, as we need to apply a transformationt to this tile content to z-up in the 3dtile file.
    
    Parameters
    ----------
    pos_list : numpy.array
    array of vertices of the gltf file must be z-up.
    
    geo_loc : tuple
    Tuple of 3 specifying the Lon/Lat/Altitude in degrees and metres.The geo-ref position of the midpt of the pos list. If the geo_loc is specified enu2fixframe transformation is applied to the bvol bbox.

    Returns
    -------
    bbox : tuple
    Tuple of 12 numbers that define the bbox for bvol of a 3dtile 
    
    """
    
    bbox = geomie3d.calculate.bbox_frm_xyzs(pos_list)
    bvol_bbox = compute_tile_bbox_frm_bbox(bbox, geo_loc = geo_loc)
    return bvol_bbox 

def compute_tile_bbox_frm_bbox(bbox, geo_loc = None):
    """
    This function only works if the gltf content is in z-up or y-up coordinate system, we are assuming the gltf has z-up coordinates while being transform to y-up in the gltf file
    in the root node of the gltf file. So we need to transform it to y-up, as we need to apply a transformationt to this tile content to z-up in the 3dtile file.
    
    Parameters
    ----------
    bbox : numpy.array
    array of 6 numbers [mnx,mny,mnz,mxx,mxy,mxz].
    
    geo_loc : tuple
    Tuple of 3 specifying the Lon/Lat/Altitude in degrees and metres.The geo-ref position of the midpt of the pos list. If the geo_loc is specified enu2fixframe transformation is applied to the bvol bbox.

    Returns
    -------
    bbox : tuple
    Tuple of 12 numbers that define the bbox for bvol of a 3dtile 
    
    """
    midpt = geomie3d.calculate.bbox_centre(bbox).tolist()
    bbox_arr = bbox.bbox_arr
    x_half_len = (bbox_arr[3] - bbox_arr[0])/2
    x_axis = [x_half_len, 0.0, 0.0]
        
    y_half_len = (bbox_arr[4] - bbox_arr[1])/2
    y_axis = [0.0, y_half_len, 0.0]
        
    z_half_len = (bbox_arr[5] - bbox_arr[2])/2
    z_axis = [0.0, 0.0, z_half_len]
    
    if geo_loc != None:
        epsg4326_4978 = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
        geo_loc_xyz = epsg4326_4978.transform(geo_loc[0], geo_loc[1], geo_loc[2])
        e2f = compute_trsfmat4enu_frm_gcs_coord(geo_loc)
        e2f_rot = np.array([[e2f[0][0], e2f[0][1], e2f[0][2], 0],
                            [e2f[1][0], e2f[1][1], e2f[1][2], 0],
                            [e2f[2][0], e2f[2][1], e2f[2][2], 0],
                            [e2f[3][0], e2f[3][1], e2f[3][2], 1]])
        
        x_axis = trsf_pos([x_axis], e2f_rot)[0]
        y_axis = trsf_pos([y_axis], e2f_rot)[0]
        z_axis = trsf_pos([z_axis], e2f_rot)[0]
        
        tx = geo_loc_xyz[0] - midpt[0]
        ty = geo_loc_xyz[1] - midpt[1]
        tz = geo_loc_xyz[2] - midpt[2]
        trsl_mat = geomie3d.calculate.translate_matrice(tx, ty, tz)     
        midpt = trsf_pos([midpt], trsl_mat)[0]
        
    midpt.extend(x_axis)
    midpt.extend(y_axis)
    midpt.extend(z_axis)
    return midpt

def compute_bregion(bbox, geo_loc):
    """
    This function only works if the gltf content is already geo-referenced to epsg:4978, we are assuming the gltf has z-up coordinates while being transform to y-up in the gltf file
    in the root node of the gltf file. So we need to transform it to y-up, as we need to apply a transformationt to this tile content to z-up in the 3dtile file.
    """
    epsg4326_4978 = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
    epsg4978_4979 = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy = True)
    
    geo_loc_xyz = epsg4326_4978.transform(geo_loc[0], geo_loc[1], geo_loc[2])
    
    mnx = bbox[0]
    mny = bbox[1]
    mnz = bbox[2]
    
    mxx = bbox[3]
    mxy = bbox[4]
    mxz = bbox[5]
    
    x_half_len = (mxx - mnx)/2
    y_half_len = (mxy - mny)/2
    z_half_len = (mxz - mnz)/2
    
    rad_list = []
    rad = True
    west = [geo_loc_xyz[0] - x_half_len, geo_loc_xyz[1], geo_loc_xyz[2]-z_half_len]
    west_mx = [geo_loc_xyz[0] - x_half_len, geo_loc_xyz[1], geo_loc_xyz[2]+z_half_len]
    west_rad = epsg4978_4979.transform(west[0], west[1], west[2], radians = rad)
    west_rad_mx = epsg4978_4979.transform(west[0], west[1], west_mx[2], radians = rad)
    rad_list.append(list(west_rad))
    rad_list.append(list(west_rad_mx))
    
    south = [geo_loc_xyz[0], geo_loc_xyz[1] - y_half_len, geo_loc_xyz[2]-z_half_len]
    south_mx = [geo_loc_xyz[0], geo_loc_xyz[1] - y_half_len, geo_loc_xyz[2]+z_half_len]
    south_rad = epsg4978_4979.transform(south[0], south[1], south[2], radians = rad)
    south_rad_mx = epsg4978_4979.transform(south[0], south[1], south_mx[2], radians = rad)
    rad_list.append(list(south_rad))
    rad_list.append(list(south_rad_mx))
    
    east = [geo_loc_xyz[0] + x_half_len, geo_loc_xyz[1], geo_loc_xyz[2]-z_half_len]
    east_mx = [geo_loc_xyz[0] - x_half_len, geo_loc_xyz[1], geo_loc_xyz[2]+z_half_len]

    east_rad = epsg4978_4979.transform(east[0], east[1], east[2], radians = rad)
    east_rad_mx = epsg4978_4979.transform(east[0], east[1], east_mx[2], radians = rad)
    rad_list.append(list(east_rad))
    rad_list.append(list(east_rad_mx))
    
    north = [geo_loc_xyz[0], geo_loc_xyz[1] + y_half_len, geo_loc_xyz[2]-z_half_len]
    north_mx = [geo_loc_xyz[0], geo_loc_xyz[1] + y_half_len, geo_loc_xyz[2]+z_half_len]
    north_rad = epsg4978_4979.transform(north[0], north[1], north[2], radians = rad)
    north_rad_mx = epsg4978_4979.transform(north[0], north[1], north_mx[2], radians = rad)
    rad_list.append(list(north_rad))
    rad_list.append(list(north_rad_mx))
    
    zip_rad = zip(*rad_list)
    z_val = list(zip_rad)[2]
    mnz_4979 = min(z_val)
    mxz_4979 = max(z_val)
    
    # print('west', west_rad)
    # print('south', south_rad)
    # print('east', east_rad)
    # print('north', north_rad)
    
    region = [west_rad[0], south_rad[1], east_rad[0], north_rad[1], mnz_4979, mxz_4979]
    return region

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
        if magic == 'i3dm':
            return I3dm.from_array(array)
        
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
#=====================================================================================================================
# endregion: Legacy code to be deprecated
#=====================================================================================================================