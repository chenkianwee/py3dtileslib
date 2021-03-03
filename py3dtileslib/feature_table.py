# -*- coding: utf-8 -*-

import json
from enum import Enum
import numpy as np


class Feature(object):

    def __init__(self):
        self.positions = {}
        self.colors = {}
        self.normal = {}
        
        self.non_uni_scale = {}
        self.scale = {}
        self.normal_up = {}
        self.normal_right = {}
        self.batch_id = {}
        
    def to_array(self):
        """
        convert all the data into a dictionary of arrays
        
        Parameters
        ----------

        Returns
        -------
        arr_dictionary : dictionary
        dictionary that contains all the arr of the feature with these keywords
        'position', 'color', 'non_uni_scale'
        """
        
        pos_arr = np.array([(self.positions['X'], self.positions['Y'],
                            self.positions['Z'])]).view(np.uint8)[0]

        if len(self.colors):
            col_arr = np.array([(self.colors['Red'], self.colors['Green'],
                                self.colors['Blue'])]).view(np.uint8)[0]
        else:
            col_arr = np.array([])
        
        
        if len(self.normal):
            nrml_arr = np.array([(self.normal['nx'], self.normal['ny'],
                                 self.normal['nz'])]).view(np.uint8)[0]
        else:
            nrml_arr = np.array([])
        
        if len(self.non_uni_scale):
            nus_arr = np.array([(self.non_uni_scale['sx'], self.non_uni_scale['sy'],
                                 self.non_uni_scale['sz'])]).view(np.uint8)[0]
        else:
            nus_arr = np.array([])
        
        if len(self.scale):
            scale_arr = np.array([(self.scale['s'])]).view(np.uint8)[0]
        else:
            scale_arr = np.array([])
        
        if len(self.normal_up):
            nrmlup_arr = np.array([(self.normal_up['nux'], self.normal_up['nuy'],
                                    self.normal_up['nuz'])]).view(np.uint8)[0]
        else:
            nrmlup_arr = np.array([])
            
        if len(self.normal_right):
            nrmlright_arr = np.array([(self.normal_right['nrx'], self.normal_right['nry'],
                                    self.normal_right['nrz'])]).view(np.uint8)[0]
        else:
            nrmlright_arr = np.array([])
            
        if len(self.batch_id):
            batch_id_arr = np.array([(self.batch_id['bid'])]).view(np.uint8)[0]
        else:
            batch_id_arr = np.array([])
            
        array_dict = {'position': pos_arr, 'color':col_arr, 'normal':nrml_arr, 'non_uni_scale': nus_arr,
                      'scale':scale_arr, 'normal_up': nrmlup_arr, 'normal_right': nrmlright_arr,
                      'batch_id': batch_id_arr}
        
        return array_dict

    @staticmethod
    def from_values(x, y, z, red=None, green=None, blue=None):
        f = Feature()
        pos_dt = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
        positions = np.array([(x,y,z)], dtype=pos_dt).view('uint8')
        
        f.positions = {}
        
        off = 0
        for d in pos_dt.names:
            dt = pos_dt[d]
            data = np.array(positions[off:off + dt.itemsize]).view(dt)[0]
            
            off += dt.itemsize
            f.positions[d] = data
        
        if red or green or blue:
            f.colors = {}
            colour_dt = np.dtype([('Red', np.uint8), ('Green', np.uint8), ('Blue', np.uint8)])
            colours = np.array([(red, green, blue)], dtype=colour_dt).view('uint8')
            
            off1 = 0
            for d1 in colour_dt.names:
                dt1 = colour_dt[d1]
                data1 = np.array(colours[off1:off1 + dt1.itemsize]).view(dt1)[0]
                
                off1 += dt1.itemsize
                f.colors[d1] = data1
            
        else:
            f.colors = {}

        return f
    
    @staticmethod
    def from_array(positions_dtype, positions, colors_dtype=None, colors=None, nrml_dtype = None, nrml = None,
                   nus_dtype = None, nus = None, scale_dtype = None, scale = None, nrmlup_dtype = None, nrml_up = None,
                   nrmlright_dtype = None, nrml_right = None, batchid_dtype = None, batchid = None):
        """
        Parameters
        ----------
        positions_dtype : numpy.dtype

        positions : numpy.array
            Array of uint8.

        colors_dtype : numpy.dtype

        colors : numpy.array
            Array of uint8.

        Returns
        -------
        f : Feature
        
        # create the numpy dtype for positions with 32-bit floating point numbers
        dt = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
        dt2 = np.dtype([('Red', np.uint8), ('Green', np.uint8), ('Blue', np.uint8)])
        position = np.array([(0,1,0)], dtype=dt)
        colour = np.array([(255,0,0)], dtype=dt2)
        
        # create a new feature from a uint8 numpy array
        f = py3dtileslib.Feature.from_array(dt, position.view('uint8'), colors_dtype=dt2, colors=col.view('uint8'))
        """

        f = Feature()

        # extract positions
        f.positions = {}
        off = 0
        for d in positions_dtype.names:
            dt = positions_dtype[d]
            data = np.array(positions[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            f.positions[d] = data

        # extract colors
        f.colors = {}
        if colors_dtype is not None:
            off = 0
            for d in colors_dtype.names:
                dt = colors_dtype[d]
                data = np.array(colors[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                f.colors[d] = data
        
        # extract nrml
        f.normal = {}
        if nrml_dtype is not None:
            off = 0
            for d in nrml_dtype.names:
                dt = nrml_dtype[d]
                data = np.array(nrml[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                
                f.normal[d] = data

        #extract nus
        f.non_uni_scale = {}
        if nus_dtype is not None:
            off = 0
            for d in nus_dtype.names:
                dt = nus_dtype[d]
                data = np.array(nus[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                
                f.non_uni_scale[d] = data
                
        #extract scale
        f.scale = {}
        if scale_dtype is not None:
            off = 0
            for d in scale_dtype.names:
                dt = scale_dtype[d]
                data = np.array(scale[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                
                f.scale[d] = data
                
        #extract nrml up
        f.normal_up = {}
        if nrmlup_dtype is not None:
            off = 0
            for d in nrmlup_dtype.names:
                dt = nrmlup_dtype[d]
                data = np.array(nrml_up[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                
                f.normal_up[d] = data
        
        f.normal_right = {}
        if nrmlright_dtype is not None:
            off = 0
            for d in nrmlright_dtype.names:
                dt = nrmlright_dtype[d]
                data = np.array(nrml_right[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                
                f.normal_right[d] = data
        
        f.batch_id = {}
        if batchid_dtype is not None:
            off = 0
            for d in batchid_dtype.names:
                dt = batchid_dtype[d]
                data = np.array(batchid[off:off + dt.itemsize]).view(dt)[0]
                off += dt.itemsize
                
                f.batch_id[d] = data
                
        return f
    
    def add_normal(self, normal):
        """
        Parameters
        ----------
        normal : numpy.array
            Array of xyz.
            
        """
        # extract normal
        nrml_dt = np.dtype([('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')])
        normal = np.array([tuple(normal)], dtype=nrml_dt).view('uint8')
        
        off = 0
        for d in nrml_dt.names:
            dt = nrml_dt[d]
            data = np.array(normal[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            
            self.normal[d] = data
            
    def add_non_uni_scale(self, non_uni_scale):
        """
        Parameters
        ----------
        non_uni_scale : tuple of 3
            tuple of sx, sy, sz
        
        """
        # extract scale
        non_uni_scale_dt = np.dtype([('sx', '<f4'), ('sy', '<f4'), ('sz', '<f4')])
        non_uni_scale = np.array([tuple(non_uni_scale)], dtype=non_uni_scale_dt).view('uint8')
        
        off = 0
        for d in non_uni_scale_dt.names:
            dt = non_uni_scale_dt[d]
            data = np.array(non_uni_scale[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            
            self.non_uni_scale[d] = data
    
    def add_scale(self, scale):
        """
        Parameters
        ----------
        scale : float

        """
        # extract scale
        scale_dt = np.dtype([('s', '<f4')])
        scale = np.array([tuple([scale])], dtype=scale_dt).view('uint8')
        
        off = 0
        for d in scale_dt.names:
            dt = scale_dt[d]
            data = np.array(scale[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            
            self.scale[d] = data
    
    def add_normal_right(self, x_axis):
        """
        Parameters
        ----------
        x_axis : numpy.array
            Array of xyz.
            
        """
        # extract scale
        x_axis_dt = np.dtype([('nux', '<f4'), ('nuy', '<f4'), ('nuz', '<f4')])
        x_axis = np.array([tuple(x_axis)], dtype=x_axis_dt).view('uint8')
        
        off = 0
        for d in x_axis_dt.names:
            dt = x_axis_dt[d]
            data = np.array(x_axis[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            
            self.normal_right[d] = data
    
    def add_normal_up(self, z_axis):
        """
        Parameters
        ----------
        z_axis : numpy.array
            Array of xyz.
            
        """
         # extract scale
        z_axis_dt = np.dtype([('nrx', '<f4'), ('nry', '<f4'), ('nrz', '<f4')])
        z_axis = np.array([tuple(z_axis)], dtype=z_axis_dt).view('uint8')
        
        off = 0
        for d in z_axis_dt.names:
            dt = z_axis_dt[d]
            data = np.array(z_axis[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            
            self.normal_up[d] = data
    
    def add_batch_id(self, batch_id):
        """
        Parameters
        ----------
        batch_id : int
            The batch id of the feature.
            
        """
        # extract scale
        batch_id_dt = np.dtype([('bid', np.uint16)])
        batch_id = np.array([tuple(batch_id)], dtype=batch_id_dt).view('uint8')
        
        off = 0
        for d in batch_id_dt.names:
            dt = batch_id_dt[d]
            data = np.array(batch_id[off:off + dt.itemsize]).view(dt)[0]
            off += dt.itemsize
            
            self.batch_id[d] = data

class SemanticPoint(Enum):

    NONE = 0
    POSITION = 1
    POSITION_QUANTIZED = 2
    RGBA = 3
    RGB = 4
    RGB565 = 5
    NORMAL = 6
    NORMAL_OCT16P = 7
    BATCH_ID = 8

class SemanticInstance(Enum):

    NONE = 0
    NORMAL_UP = 1
    NORMAL_RIGHT = 2
    SCALE = 3
    NON_UNI_SCALE = 4
    BATCHID = 5

class FeatureTableHeader(object):

    def __init__(self):
        # point semantics
        self.positions = SemanticPoint.NONE
        self.positions_offset = 0
        self.positions_dtype = None

        self.colors = SemanticPoint.NONE
        self.colors_offset = 0
        self.colors_dtype = None

        self.normal = SemanticPoint.NONE
        self.normal_offset = 0
        self.normal_dtype = None

        # global semantics
        self.points_length = None
        self.rtc = None
        
        #b3dm semantics
        self.batch_length = None
        
        #i3dm semantics
        self.non_uniform_scale = SemanticInstance.NONE
        self.nus_offset = 0
        self.nus_dtype = None
        
        self.scale = SemanticInstance.NONE
        self.scale_offset = 0
        self.scale_dtype = None
        
        self.normal_up = SemanticInstance.NONE
        self.normal_up_offset = 0
        self.normal_up_dtype = None
        
        self.normal_right = SemanticInstance.NONE
        self.normal_right_offset = 0
        self.normal_right_dtype = None
        
        self.batch_id = SemanticInstance.NONE
        self.batch_id_offset = 0
        self.batch_id_dtype = None
        
        #i3dm semantics global semantics
        self.instances_length = None
        self.east_north_up = None
        

    def to_array(self):
        jsond = self.to_json()
        json_str = json.dumps(jsond).replace(" ", "")
        n = len(json_str) + 28
        json_str += ' ' * (4 - n % 4)
        # return np.frombuffer(json_str.encode('utf-8'), dtype=np.uint8)
        return np.fromstring(json_str, dtype=np.uint8)
        
    def to_json(self):
        jsond = {}

        # length
        if self.points_length != 0:
            jsond['POINTS_LENGTH'] = self.points_length

        # rtc
        if self.rtc:
            jsond['RTC_CENTER'] = self.rtc

        # positions
        if self.positions == SemanticPoint.POSITION:
            offset = {'byteOffset': self.positions_offset}
            if self.positions == SemanticPoint.POSITION:
                jsond['POSITION'] = offset
            elif self.positions == SemanticPoint.POSITION_QUANTIZED:
                jsond['POSITION_QUANTIZED'] = offset

        # colors
        if self.colors == SemanticPoint.RGB:
            offset = {'byteOffset': self.colors_offset}
            jsond['RGB'] = offset
        
        # normals
        if self.normal == SemanticPoint.NORMAL:
            offset = {'byteOffset': self.normal_offset}
            jsond['NORMAL'] = offset
        
        #non uniform scale 
        if self.non_uniform_scale == SemanticInstance.NON_UNI_SCALE:
            offset = {'byteOffset': self.nus_offset}
            jsond['SCALE_NON_UNIFORM'] = offset
        
        #scale 
        if self.scale == SemanticInstance.SCALE:
            offset = {'byteOffset': self.scale_offset}
            jsond['SCALE'] = offset
        
        #normal up  
        if self.normal_up == SemanticInstance.NORMAL_UP:
            offset = {'byteOffset': self.normal_up_offset}
            jsond['NORMAL_UP'] = offset
        
        #normal right  
        if self.normal_right == SemanticInstance.NORMAL_RIGHT:
            offset = {'byteOffset': self.normal_right_offset}
            jsond['NORMAL_RIGHT'] = offset
        
        #normal batchid
        if self.batch_id == SemanticInstance.BATCHID:
            offset = {'byteOffset': self.batch_id_offset}
            jsond['BATCH_ID'] = offset
            
        #batch_length
        if self.batch_length:
            jsond['BATCH_LENGTH'] = self.batch_length
                    
        return jsond

    @staticmethod
    def from_dtype(positions_dtype, nfeatures, ft_type, colors_dtype = None, normal_dtype = None, 
                   non_uni_scale_dtype = None, scale_dtype = None, nrmlup_dtype = None, 
                   nrmlright_dtype = None, batchid_dtype = None):
        """
        Parameters
        ----------
        positions_dtype : numpy.dtype
            Numpy description of a positions.

        colors_dtype : numpy.dtype
            Numpy description of a colors.

        Returns
        -------
        fth : FeatureTableHeader
        """

        fth = FeatureTableHeader()
        if ft_type == 'pnts':
            fth.points_length = nfeatures
        elif ft_type == 'pnts':
            fth.instances_length = nfeatures
        
        pos_size = 0
        color_size = 0
        normal_size = 0
        nus_size = 0
        scale_size = 0
        nrmlup_size = 0
        nrmlright_size = 0
        # batchid_size = 0
        
        # search positions
        names = positions_dtype.names
        if ('X' in names) and ('Y' in names) and ('Z' in names):
            dtx = positions_dtype['X']
            dty = positions_dtype['Y']
            dtz = positions_dtype['Z']
            fth.positions_offset = 0
            if (dtx == np.float32 and dty == np.float32 and dtz == np.float32):
                fth.positions = SemanticPoint.POSITION
                fth.positions_dtype = np.dtype([('X', np.float32),
                                                ('Y', np.float32),
                                                ('Z', np.float32)])
            elif (dtx == np.uint16 and dty == np.uint16 and dtz == np.uint16):
                fth.positions = SemanticPoint.POSITION_QUANTIZED
                fth.positions_dtype = np.dtype([('X', np.uint16),
                                                ('Y', np.uint16),
                                                ('Z', np.uint16)])
                
            pos_size = nfeatures * fth.positions_dtype.itemsize

        # search colors
        if colors_dtype is not None:
            names = colors_dtype.names
            if ('Red' in names) and ('Green' in names) and ('Blue' in names):
                if 'Alpha' in names:
                    fth.colors = SemanticPoint.RGBA
                    fth.colors_dtype = np.dtype([('Red', np.uint8),
                                                 ('Green', np.uint8),
                                                 ('Blue', np.uint8),
                                                 ('Alpha', np.uint8)])
                else:
                    fth.colors = SemanticPoint.RGB
                    fth.colors_dtype = np.dtype([('Red', np.uint8),
                                                 ('Green', np.uint8),
                                                 ('Blue', np.uint8)])

                fth.colors_offset = (fth.positions_offset
                                     + pos_size)
                color_size = nfeatures * fth.colors_dtype.itemsize
                
        #search for normals
        if normal_dtype is not None:
            names = normal_dtype.names
            if ('nx' in names) and ('ny' in names) and ('nz' in names):
                fth.normal = SemanticPoint.NORMAL
                fth.normal_dtype = np.dtype([('nx', np.float32),
                                             ('ny', np.float32),
                                             ('nz', np.float32)])
                    
                fth.normal_offset = (fth.positions_offset
                                     + pos_size
                                     + color_size)
                
                normal_size = nfeatures * fth.normal_dtype.itemsize
                
        #search for non_uniform_scale
        if non_uni_scale_dtype is not None:
            names = non_uni_scale_dtype.names
            if ('sx' in names) and ('sy' in names) and ('sz' in names):
                fth.non_uniform_scale = SemanticInstance.NON_UNI_SCALE
                fth.nus_dtype = np.dtype([('sx', np.float32),
                                          ('sy', np.float32),
                                          ('sz', np.float32)])
                    
                fth.nus_offset = (fth.positions_offset
                                  + pos_size
                                  + color_size
                                  + normal_size)
                
                nus_size = nfeatures * fth.nus_dtype.itemsize
                
        #search for scale
        if scale_dtype is not None:
            names = scale_dtype.names
            if ('s' in names):
                fth.scale = SemanticInstance.SCALE
                fth.scale_dtype = np.dtype([('s', np.float32)])
                fth.scale_offset = (fth.positions_offset
                                    + pos_size
                                    + color_size
                                    + normal_size
                                    + nus_size)
                
                scale_size = nfeatures * fth.scale_dtype.itemsize
        
        #search for nrmlup
        if nrmlup_dtype is not None:
            names = nrmlup_dtype.names
            if ('nux' in names) and ('nuy' in names) and ('nuz' in names):
                fth.normal_up = SemanticInstance.NORMAL_UP
                fth.normal_up_dtype = np.dtype([('nux', np.float32),
                                                ('nuy', np.float32),
                                                ('nuz', np.float32)])
                    
                fth.normal_up_offset = (fth.positions_offset
                                        + pos_size
                                        + color_size
                                        + normal_size
                                        + nus_size
                                        + scale_size)
                
                nrmlup_size = nfeatures * fth.normal_up_dtype.itemsize
                
        #search for nrmlright
        if nrmlright_dtype is not None:
            names = nrmlright_dtype.names
            if ('nrx' in names) and ('nry' in names) and ('nrz' in names):
                fth.normal_right = SemanticInstance.NORMAL_RIGHT
                fth.normal_right_dtype = np.dtype([('nrx', np.float32),
                                                     ('nry', np.float32),
                                                     ('nrz', np.float32)])
                    
                fth.normal_right_offset = (fth.positions_offset
                                           + pos_size
                                           + color_size
                                           + normal_size
                                           + nus_size
                                           + scale_size
                                           + nrmlup_size)
                
                nrmlright_size = nfeatures * fth.normal_right_dtype.itemsize
        
        if nrmlup_dtype == None and nrmlright_dtype == None:
            fth.east_north_up = True
        
        #search for batch id
        if batchid_dtype is not None:
            names = batchid_dtype.names
            if ('bid' in names):
                fth.batch_id = SemanticInstance.BATCHID
                fth.batch_id_dtype = np.dtype([('bid', np.uint16)])
                fth.batch_id_offset = (fth.positions_offset
                                      + pos_size
                                      + color_size
                                      + normal_size
                                      + nus_size
                                      + scale_size
                                      + nrmlup_size
                                      + nrmlright_size)
                
                # batchid_size = nfeatures * fth.batchid_dtype.itemsize
                
        return fth

    @staticmethod
    def from_array(array):
        """
        Parameters
        ----------
        array : numpy.array
            Json in 3D Tiles format. See py3dtiles/doc/semantics.json for an
            example.

        Returns
        -------
        fth : FeatureTableHeader
        """

        jsond = json.loads(array.tobytes().decode('utf-8'))
        fth = FeatureTableHeader()

        # search position
        if "POSITION" in jsond:
            fth.positions = SemanticPoint.POSITION
            fth.positions_offset = jsond['POSITION']['byteOffset']
            fth.positions_dtype = np.dtype([('X', np.float32),
                                            ('Y', np.float32),
                                            ('Z', np.float32)])
        elif "POSITION_QUANTIZED" in jsond:
            fth.positions = SemanticPoint.POSITION_QUANTIZED
            fth.positions_offset = jsond['POSITION_QUANTIZED']['byteOffset']
            fth.positions_dtype = np.dtype([('X', np.uint16),
                                            ('Y', np.uint16),
                                            ('Z', np.uint16)])
        else:
            fth.positions = SemanticPoint.NONE
            fth.positions_offset = 0
            fth.positions_dtype = None

        # search colors
        if "RGB" in jsond:
            fth.colors = SemanticPoint.RGB
            fth.colors_offset = jsond['RGB']['byteOffset']
            fth.colors_dtype = np.dtype([('Red', np.uint8),
                                         ('Green', np.uint8),
                                         ('Blue', np.uint8)])
        else:
            fth.colors = SemanticPoint.NONE
            fth.colors_offset = 0
            fth.colors_dtype = None
        
        
        #search normals
        if "NORMAL" in jsond:
            fth.normal = SemanticPoint.NORMAL
            fth.normal_offset = jsond['RGB']['byteOffset']
            fth.normal_dtype = np.dtype([('nx', np.float32),
                                         ('ny', np.float32),
                                         ('nz', np.float32)])
        else:
            fth.normal = SemanticPoint.NONE
            fth.normal_offset = 0
            fth.normal_dtype = None
            
        #search non uniform scale
        if "SCALE_NON_UNIFORM" in jsond:
            fth.non_uniform_scale = SemanticInstance.NON_UNI_SCALE
            fth.nus_offset = jsond['SCALE_NON_UNIFORM']['byteOffset']
            fth.nus_dtype = np.dtype([('sx', np.float32),
                                      ('sy', np.float32),
                                      ('sz', np.float32)])
        else:
            fth.non_uniform_scale = SemanticInstance.NONE
            fth.nus_offset = 0
            fth.nus_dtype = None
        
        #search for scale
        if "SCALE" in jsond:
            fth.scale = SemanticInstance.SCALE
            fth.scale_offset = jsond['SCALE_NON_UNIFORM']['byteOffset']
            fth.scale_dtype = np.dtype([('s', np.float32)])
        else:
            fth.scale = SemanticInstance.NONE
            fth.scale_offset = 0
            fth.scale_dtype = None
            
        #search for normal up
        if "NORMAL_UP" in jsond:
            fth.normal_up = SemanticInstance.NORMAL_UP
            fth.normal_up_offset = jsond['NORMAL_UP']['byteOffset']
            fth.normal_up_dtype = np.dtype([('nux', np.float32),
                                            ('nuy', np.float32),
                                            ('nuz', np.float32)])
        else:
            fth.normal_up = SemanticInstance.NONE
            fth.normal_up_offset = 0
            fth.normal_up_dtype = None
        
        #search for normal right
        if "NORMAL_RIGHT" in jsond:
            fth.normal_right = SemanticInstance.NORMAL_RIGHT
            fth.normal_right_offset = jsond['NORMAL_RIGHT']['byteOffset']
            fth.normal_right_dtype = np.dtype([('nrx', np.float32),
                                               ('nry', np.float32),
                                               ('nrz', np.float32)])
        else:
            fth.normal_right = SemanticInstance.NONE
            fth.normal_right_offset = 0
            fth.normal_right_dtype = None
        
        #search for batch id 
        if "BATCH_ID" in jsond:
            fth.batch_id = SemanticInstance.BATCHID
            fth.batch_id_offset = jsond['BATCH_ID']['byteOffset']
            fth.batch_id_dtype = np.dtype([('bid', np.uint16)])
        else:
            fth.batch_id = SemanticInstance.NONE
            fth.batch_id_offset = 0
            fth.batch_id_dtype = None
            
        # points length
        if "POINTS_LENGTH" in jsond:
            fth.points_length = jsond["POINTS_LENGTH"]
        else:
            fth.points_length = None
            
        #instance length
        if "INSTANCES_LENGTH" in jsond:
            fth.instances_length = jsond["INSTANCES_LENGTH"]
        else:
            fth.instances_length = None

        # RTC (Relative To Center)
        if "RTC_CENTER" in jsond:
            fth.rtc = jsond['RTC_CENTER']
        else:
            fth.rtc = None
        
        # BATCH LENGTH
        if "BATCH_LENGTH" in jsond:
            fth.batch_length = jsond['BATCH_LENGTH']
        else:
            fth.batch_length = None
        
        # EAST NORTH UP
        if "EAST_NORTH_UP" in jsond:
            fth.east_north_up = jsond['EAST_NORTH_UP']
        else:
            fth.east_north_up = None
        
        return fth

class FeatureTableBody(object):

    def __init__(self):
        self.positions_arr = []
        self.positions_itemsize = 0

        self.colors_arr = []
        self.colors_itemsize = 0
        
        self.normal_arr = []
        self.normal_itemsize = 0
        
        self.nus_arr = []
        self.nus_itemsize = 0
        
        self.scale_arr = []
        self.scale_itemsize = 0
        
        self.normal_up_arr = []
        self.normal_up_itemsize = 0
        
        self.normal_right_arr = []
        self.normal_right_itemsize = 0
        
        self.batch_id_arr = []
        self.batch_id_itemsize = 0        

    def to_array(self):
        arr = self.positions_arr
        
        if len(self.colors_arr):
            arr = np.concatenate((self.positions_arr, self.colors_arr))
            
        if len(self.normal_arr):
            arr = np.concatenate((arr, self.normal_arr))
            
        if len(self.nus_arr):
            arr = np.concatenate((arr, self.nus_arr))
        
        if len(self.scale_arr):
            arr = np.concatenate((arr, self.scale_arr))
            
        if len(self.normal_up_arr):
            arr = np.concatenate((arr, self.normal_up_arr))
        
        if len(self.normal_right_arr):
            arr = np.concatenate((arr, self.normal_right_arr))
        
        if len(self.batch_id_arr):
            arr = np.concatenate((arr, self.batch_id_arr))
        
        return arr

    @staticmethod
    def from_features(fth, features):

        b = FeatureTableBody()

        # extract positions
        if fth.positions_dtype != None:
            b.positions_itemsize = fth.positions_dtype.itemsize
            
        b.positions_arr = np.array([], dtype=np.uint8)

        #extract colours 
        if fth.colors_dtype is not None:
            b.colors_itemsize = fth.colors_dtype.itemsize
            b.colors_arr = np.array([], dtype=np.uint8)
        
        #extract normal  
        if fth.normal_dtype is not None:
            b.normal_itemsize = fth.normal_dtype.itemsize
            b.normal_arr = np.array([], dtype=np.uint8)
        
        #extract non uniform scale  
        if fth.nus_dtype is not None:
            b.nus_itemsize = fth.nus_dtype.itemsize
            b.nus_arr = np.array([], dtype=np.uint8)
            
        #extract scale  
        if fth.scale_dtype is not None:
            b.scale_itemsize = fth.scale_dtype.itemsize
            b.scale_arr = np.array([], dtype=np.uint8)
        
        #extract normal up  
        if fth.normal_up_dtype is not None:
            b.normal_up_itemsize = fth.normal_up_dtype.itemsize
            b.normal_up_arr = np.array([], dtype=np.uint8)
        
        #extract normal right
        if fth.normal_right_dtype is not None:
            b.normal_right_itemsize = fth.normal_right_dtype.itemsize
            b.normal_right_arr = np.array([], dtype=np.uint8)
        
        #extract batch_id
        if fth.batch_id_dtype is not None:
            b.batch_id_itemsize = fth.batch_id_dtype.itemsize
            b.batch_id_arr = np.array([], dtype=np.uint8)
        
        for f in features:
            arr_d = f.to_array()
            fpos = arr_d['position']
            fcol = arr_d['color']
            fn = arr_d['normal']
            fnus = arr_d['non_uni_scale']
            fsc = arr_d['scale']
            fnu =  arr_d['normal_up']
            fnr = arr_d['normal_right']
            fbid = arr_d['batch_id']
            
            b.positions_arr = np.concatenate((b.positions_arr, fpos))
            
            if fth.colors_dtype is not None:
                b.colors_arr = np.concatenate((b.colors_arr, fcol))
            
            if fth.normal_dtype is not None:
                b.normal_arr = np.concatenate((b.normal_arr, fn))
            
            if fth.nus_dtype is not None:
                b.nus_arr = np.concatenate((b.nus_arr, fnus))
                
            if fth.scale_dtype is not None:
                b.scale_arr = np.concatenate((b.scale_arr, fsc))
            
            if fth.normal_up_dtype is not None:
                b.normal_up_arr = np.concatenate((b.normal_up_arr, fnu))
                
            if fth.normal_right_dtype is not None:
                b.normal_right_arr = np.concatenate((b.normal_right_arr, fnr))
            
            if fth.batch_id_dtype is not None:
                b.batch_id_arr = np.concatenate((b.batch_id_arr, fbid))

        return b

    @staticmethod
    def from_array(fth, array):
        """
        Parameters
        ----------
        header : FeatureTableHeader

        array : numpy.array

        Returns
        -------
        ftb : FeatureTableBody
        """

        b = FeatureTableBody()
        
        if fth.points_length == None:
            nfeatures = fth.instances_length
        else:
            nfeatures = fth.points_length

        # extract positions
        if fth.positions != SemanticPoint.NONE:
            pos_size = fth.positions_dtype.itemsize
            pos_offset = fth.positions_offset
            b.positions_arr = array[pos_offset:pos_offset + nfeatures * pos_size]
            b.positions_itemsize = pos_size

        # extract colors
        if fth.colors != SemanticPoint.NONE:
            col_size = fth.colors_dtype.itemsize
            col_offset = fth.colors_offset
            b.colors_arr = array[col_offset:col_offset + nfeatures * col_size]
            b.colors_itemsize = col_size
        
        # extract normal
        if fth.normal != SemanticPoint.NONE:
            nrml_size = fth.normal_dtype.itemsize
            nrml_offset = fth.normal_offset
            b.normal_arr = array[nrml_offset:nrml_offset + nfeatures * nrml_size]
            b.normal_itemsize = nrml_size
        
        # extract non uniform scale
        if fth.non_uniform_scale != SemanticInstance.NONE:
            nus_size = fth.nus_dtype.itemsize
            nus_offset = fth.nus_offset
            b.nus_arr = array[nus_offset:nus_offset + nfeatures * nus_size]
            b.nus_itemsize = nrml_size
        
        # extract scale
        if fth.scale != SemanticInstance.NONE:
            scale_size = fth.scale_dtype.itemsize
            scale_offset = fth.scale_offset
            b.scale_arr = array[scale_offset:scale_offset + nfeatures * scale_size]
            b.scale_itemsize = scale_size
        
        # extract normal up 
        if fth.normal_up != SemanticInstance.NONE:
            nrmlup_size = fth.normal_up_dtype.itemsize
            nrmlup_offset = fth.normal_up_offset
            b.normal_up_arr = array[nrmlup_offset:nrmlup_offset + nfeatures * nrmlup_size]
            b.normal_up_itemsize = nrml_size
        
        # extract normal right 
        if fth.normal_right != SemanticInstance.NONE:
            nrmlright_size = fth.normal_right_dtype.itemsize
            nrmlright_offset = fth.normal_right_offset
            b.normal_right_arr = array[nrmlright_offset:nrmlright_offset + nfeatures * nrmlright_size]
            b.normal_right_itemsize = nrmlright_size
        
        # extract batch id
        if fth.batch_id != SemanticInstance.NONE:
            batchid_size = fth.batch_id_dtype.itemsize
            batchid_offset = fth.batch_id_offset
            b.batch_id_arr = array[batchid_offset:batchid_offset + nfeatures * batchid_size]
            b.batch_id_itemsize = batchid_size
        
        return b

    def positions(self, n):
        itemsize = self.positions_itemsize
        return self.positions_arr[n * itemsize:(n + 1) * itemsize]

    def colors(self, n):
        if len(self.colors_arr):
            itemsize = self.colors_itemsize
            return self.colors_arr[n * itemsize:(n + 1) * itemsize]
        return []
    
    def normals(self, n):
        if len(self.normal_arr):
            itemsize = self.normal_itemsize
            return self.normal_arr[n * itemsize:(n + 1) * itemsize]
        return []
    
    def non_uni_scale(self, n):
        if len(self.nus_arr):
            itemsize = self.nus_itemsize
            return self.nus_arr[n * itemsize:(n + 1) * itemsize]
        return []
    
    def scale(self, n):
        if len(self.scale_arr):
            itemsize = self.scale_itemsize
            return self.scale_arr[n * itemsize:(n + 1) * itemsize]
        return []
    
    def normal_up(self, n):
        if len(self.normal_up_arr):
            itemsize = self.normal_up_itemsize
            return self.normal_up_arr[n * itemsize:(n + 1) * itemsize]
        return []
    
    def normal_right(self, n):
        if len(self.normal_right_arr):
            itemsize = self.normal_right_itemsize
            return self.normal_right_arr[n * itemsize:(n + 1) * itemsize]
        return []

    def batch_id(self, n):
        if len(self.batch_id_arr):
            itemsize = self.batch_id_itemsize
            return self.batch_id_arr[n * itemsize:(n + 1) * itemsize]
        return []

class FeatureTable(object):

    def __init__(self):
        self.header = FeatureTableHeader()
        self.body = FeatureTableBody()

    def npoints(self):
        return self.header.points_length

    def to_array(self):
        fth_arr = self.header.to_array()
        ftb_arr = self.body.to_array()
        if len(ftb_arr) == 0:
            return fth_arr
        else:
            return np.concatenate((fth_arr, ftb_arr))

    @staticmethod
    def from_array(th, array):
        """
        Parameters
        ----------
        th : TileHeader

        array : numpy.array

        Returns
        -------
        ft : FeatureTable
        """

        # build feature table header
        fth_len = th.ft_json_byte_length
        fth_arr = array[0:fth_len]
        fth = FeatureTableHeader.from_array(fth_arr)

        # build feature table body
        ftb_len = th.ft_bin_byte_length
        ftb_arr = array[fth_len:fth_len + ftb_len]
        ftb = FeatureTableBody.from_array(fth, ftb_arr)

        # build feature table
        ft = FeatureTable()
        ft.header = fth
        ft.body = ftb

        return ft

    @staticmethod
    def from_features(features, ft_type, pdtype = None, cdtype = None, normal_dtype = None,
                      nus_dtype = None, scale_dtype = None, nrmlup_dtype = None, 
                      nrmlright_dtype = None, batchid_dtype = None):
        """
        features: Features object
            features of each point or instances
            
        ft_type: str
            Either 'pnts' or 'i3dm'
            
        pdtype : numpy.dtype
            Numpy description for positions.

        cdtype : numpy.dtype
            Numpy description for colors.

        Returns
        -------
        ft : FeatureTable
        """
        
        if pdtype == None:
            pdtype = np.dtype([('X', '<f4'), ('Y', '<f4'), ('Z', '<f4')])
            
        if cdtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['color']) > 0: 
                cdtype = np.dtype([('Red', np.uint8), ('Green', np.uint8), ('Blue', np.uint8)])
                
        if normal_dtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['normal']) > 0: 
                normal_dtype = np.dtype([('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')])
                
        if nus_dtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['non_uni_scale']) > 0: 
                nus_dtype = np.dtype([('sx', '<f4'), ('sy', '<f4'), ('sz', '<f4')])
        
        if scale_dtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['scale']) > 0: 
                nus_dtype = np.dtype([('s', '<f4')])
        
        
        if nrmlup_dtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['normal_up']) > 0: 
                nrmlup_dtype = np.dtype([('nux', '<f4'), ('nuy', '<f4'), ('nuz', '<f4')])
        
        if nrmlright_dtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['normal_right']) > 0: 
                nrmlright_dtype = np.dtype([('nrx', '<f4'), ('nry', '<f4'), ('nrz', '<f4')])
        
        if batchid_dtype == None:
            arr_dict = features[0].to_array()
            if len(arr_dict['batch_id']) > 0: 
                batchid_dtype = np.dtype([('bid', np.uint16)])
                
        
        fth = FeatureTableHeader.from_dtype(pdtype, len(features), ft_type = ft_type, colors_dtype = cdtype, normal_dtype = normal_dtype, 
                                            non_uni_scale_dtype = nus_dtype, scale_dtype = scale_dtype, nrmlup_dtype = nrmlup_dtype, 
                                            nrmlright_dtype = nrmlright_dtype, batchid_dtype = batchid_dtype)
        
        ftb = FeatureTableBody.from_features(fth, features)

        ft = FeatureTable()
        ft.header = fth
        ft.body = ftb

        return ft

    def feature(self, n):
        pos = self.body.positions(n)
        col = self.body.colors(n)
        nrml = self.body.normals(n)
        nus = self.body.non_uni_scale(n)
        scale = self.body.scale(n)
        nrml_up = self.body.normal_up(n)
        nrml_right = self.body.normal_right(n)
        batchid = self.body.batch_id(n)
        
        f = Feature.from_array(self.header.positions_dtype, pos,
                               self.header.colors_dtype, col)
        
        f = Feature.from_array(self.header.positions_dtype, pos, 
                               colors_dtype=self.header.colors_dtype, colors=col, 
                               nrml_dtype = self.header.normal_dtype, nrml = nrml,
                               nus_dtype = self.header.nus_dtype, nus = nus, 
                               scale_dtype = self.header.scale_dtype, scale = scale, 
                               nrmlup_dtype = self.header.normal_up_dtype, nrml_up = nrml_up,
                               nrmlright_dtype = self.header.normal_right_dtype, nrml_right = nrml_right, 
                               batchid_dtype = self.header.batch_id_dtype, batchid = batchid)
        
        return f
    
    def add_batch_length(self, batch_length):
        self.header.batch_length = batch_length
    
    def add_rtc(self, rtc):
        self.header.rtc = rtc
