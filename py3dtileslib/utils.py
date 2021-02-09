# -*- coding: utf-8 -*-
import os
import math
import tempfile
import numpy as np
from pyproj import CRS, Transformer
from pygltflib import GLTF2

from .pnts import Pnts
from .b3dm import B3dm

import geomie3d

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
    x_half_len = (bbox[3] - bbox[0])/2
    x_axis = [x_half_len, 0, 0]
        
    y_half_len = (bbox[4] - bbox[1])/2
    y_axis = [0, y_half_len, 0]
        
    z_half_len = (bbox[5] - bbox[2])/2
    z_axis = [0, 0, z_half_len]
    
    epsg4326_4978 = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
    
    if geo_loc != None:
        e2f = compute_enu2fixframe(geo_loc)
        e2f_rot = np.array([[e2f[0][0], e2f[0][1], e2f[0][2], 0],
                            [e2f[1][0], e2f[1][1], e2f[1][2], 0],
                            [e2f[2][0], e2f[2][1], e2f[2][2], 0],
                            [e2f[3][0], e2f[3][1], e2f[3][2], 1]])
        
        x_axis = trsf_pos([x_axis], e2f_rot)[0]
        y_axis = trsf_pos([y_axis], e2f_rot)[0]
        z_axis = trsf_pos([z_axis], e2f_rot)[0]
        
        geo_loc_xyz = epsg4326_4978.transform(geo_loc[0], geo_loc[1], geo_loc[2])
        tx = geo_loc_xyz[0] - midpt[0]
        ty = geo_loc_xyz[1] - midpt[1]
        tz = geo_loc_xyz[2] - midpt[2]
        trsl_mat = geomie3d.calculate.translate_matrice(tx, ty, tz)     
        midpt = trsf_pos([midpt], trsl_mat)[0]
        
    midpt.extend(x_axis)
    midpt.extend(y_axis)
    midpt.extend(z_axis)
    return midpt
    
def compute_enu2fixframe(geo_loc):
    """
    Compute the East North Up (ENU) coordinate to ECEF 4x4 transformation
    https://gis.stackexchange.com/questions/308445/local-enu-point-of-interest-to-ecef
    
    Parameters
    ----------
    geo_loc : tuple
    Tuple of 3 specifying the Lon/Lat/Altitude in degrees and metres.

    Returns
    -------
    trsf_mat : np.array
    2 dimnsional matrix of 4x4 transformation
    
    """
    lla_origin = geo_loc
    epsg4326_4978 = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
    ecef_origin = epsg4326_4978.transform(lla_origin[0], lla_origin[1], lla_origin[2])
        
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

def trsf_pos(pos_list, trsf_mat):
    """
    Transform the positions according to the transformation matrix
    
    Parameters
    ----------
    pos_list : numpy.array
    array of vertices of the gltf file must be z-up.
    
    trsf_mat : np.array
    2 dimnsional matrix of 4x4 transformation

    Returns
    -------
    trsf_pos_list : numpy.array
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
    upr_right =  [bbox[3], bbox[4], bbox[5]]
    error = geomie3d.calculate.dist_btw_xyzs(lwr_left, upr_right)
    return error

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