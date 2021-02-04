import os
import json
import math
import numpy as np
import geomie3d
import pyproj

class BoundingBox():
    def __init__(self, minimum, maximum):
        self.min = [float(i) for i in minimum]
        self.max = [float(i) for i in maximum]

    def inside(self, point):
        return ((self.min[0] <= point[0] < self.max[0])
                and (self.min[1] <= point[1] < self.max[1]))

    def center(self):
        return [(i + j) / 2 for (i, j) in zip(self.min, self.max)]

    def add(self, box):
        self.min = [min(i, j) for (i, j) in zip(self.min, box.min)]
        self.max = [max(i, j) for (i, j) in zip(self.max, box.max)]
        
class Tileset(object):
    def __init__(self, folder_path):
        self.properties = None
        self.error = None
        self.root = None
        self.asset = {'version':'1.0'}
        self.folder_path = folder_path
        
    def add_root(self, node):
        """
        Parameters
        ----------
        node : Node
            Node object
        """
        self.root = node
    
    def compute_error(self):
        """
        This function only works if bbox is already computed
        """
        bbox = self.root.bbox
        lwr_left = [bbox.min[0], bbox.min[1], bbox.min[2]]
        upr_right =  [bbox.max[0], bbox.max[1], bbox.min[2]]
        dist = geomie3d.calculate.dist_btw_xyzs(lwr_left, upr_right)
        self.error = dist
        
    def add_error(self, error):
        """
        Parameters
        ----------
        error : Float
            Geometric error of the tileset
        """
        self.error = error
    
    def to_tileset(self):
        tile_path = os.path.join(self.folder_path, 'tiles')
        if not os.path.isdir(tile_path):
            os.makedirs(tile_path)
            
        tiles = {
            'asset': self.asset,
            'geometricError': self.error,
            "root": self.root.to_dict(tile_path)
        }
        
        tileset_path = os.path.join(self.folder_path, 'tileset.json')
        tileset_file = open(tileset_path, 'w')
        json.dump(tiles, tileset_file)
        tileset_file.close()

        return tiles
                
class Node(object):
    def __init__(self, uniqid):
        """
        Parameters
        ----------
        uniqid : int or str
            Id that is unique to this node.
        """
        self.geo_loc = None # tuple of lon/lat/alt of the position of the gltf. This location correspond to the origin of the gltf model  
        self.transform = None
        self.bbox = None
        self.bvol = None
        self.error = None
        self.refine = 'ADD' #other options include "REPLACE"
        self.tile_content = None #TileContent object B3DM PNT I3DM
        self.children = []
        self.uniqid = uniqid
        
    def add_tile_content(self, tile_content):
        """
        Parameters
        ----------
        tile_content : TileContent object
            TileContent object B3DM PNT I3DM
        """
        self.tile_content = tile_content
    
    def add_geo_loc(self, geo_loc):
        self.geo_loc = geo_loc
    
    def compute_bbox(self):
        """
        This function only works if the gltf content is already geo-referenced to epsg:4978, we are assuming the gltf has z-up coordinates while being transform to y-up in the gltf file
        in the root node of the gltf file. So we need to transform it to y-up, as we need to apply a transformationt to this tile content to z-up in the 3dtile file.
        """
        self.bbox = BoundingBox(
            [float("inf"), float("inf"), float("inf")],
            [-float("inf"), -float("inf"), -float("inf")])
        
        if self.tile_content != None:
            if self.tile_content.header.magic_value == b"b3dm":
                gltf = self.tile_content.body.glTF
                accessors = gltf.accessors
                meshes = gltf.meshes
                for mesh in meshes:
                    prims = mesh.primitives
                    for prim in prims:
                        pos = prim.attributes.POSITION
                        pos_access = accessors[pos]
                        mx = pos_access.max
                        mn = pos_access.min
                        mn_mx = np.array([mn,mx])
                        content_bbox = BoundingBox(mn_mx[0], mn_mx[1])
                        self.bbox.add(content_bbox)
        
        for c in self.children:
            c.compute_bbox()
            self.bbox.add(c.bbox)
        
        #change to y-up
        rot_mat = geomie3d.calculate.rotate_matrice([1,0,0], -90)
        
        midpt = self.bbox.center()
        midpt = [midpt[0],midpt[1], midpt[2]]
        midpt = self.trsf_pos([midpt], rot_mat)[0]
        
        x_half_len = (self.bbox.max[0] - self.bbox.min[0])/2
        x_axis = [x_half_len, 0, 0]
        x_axis = self.trsf_pos([x_axis], rot_mat)[0]
        
        y_half_len = (self.bbox.max[1] - self.bbox.min[1])/2
        y_axis = [0, y_half_len, 0]
        y_axis = self.trsf_pos([y_axis], rot_mat)[0]
        
        z_half_len = (self.bbox.max[2] - self.bbox.min[2])/2
        z_axis = [0, 0, z_half_len]
        z_axis = self.trsf_pos([z_axis], rot_mat)[0]
        
        midpt.extend(x_axis)
        midpt.extend(y_axis)
        midpt.extend(z_axis)
        
        self.bvol = {'box':midpt}
        
    def compute_bregion(self):
        """
        This function only works if the gltf content is already geo-referenced to epsg:4978, we are assuming the gltf has z-up coordinates while being transform to y-up in the gltf file
        in the root node of the gltf file. So we need to transform it to y-up, as we need to apply a transformationt to this tile content to z-up in the 3dtile file.
        """
        zup = False
        self.bbox = BoundingBox(
            [float("inf"), float("inf"), float("inf")],
            [-float("inf"), -float("inf"), -float("inf")])
        
        if self.tile_content != None:
            if self.tile_content.header.magic_value == b"b3dm":
                gltf = self.tile_content.body.glTF
                accessors = gltf.accessors
                meshes = gltf.meshes
                nodes = gltf.nodes
                for node in nodes:
                    mat = node.matrix
                    if mat != None:
                        if mat == [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]:
                            #the original coordinate of the gltf model is assumed to be z-up
                            zup = True
                    
                for mesh in meshes:
                    prims = mesh.primitives
                    for prim in prims:
                        pos = prim.attributes.POSITION
                        pos_access = accessors[pos]
                        mx = pos_access.max
                        mn = pos_access.min
                        mn_mx = np.array([mn,mx])
                        content_bbox = BoundingBox(mn_mx[0], mn_mx[1])
                        self.bbox.add(content_bbox)
        else:
            zup = True
        
        for c in self.children:
            c.compute_bregion()
            self.bbox.add(c.bbox)
        
        epsg4326_4978 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
        epsg4978_4979 = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy = True)
        
        mnx = self.bbox.min[0]
        mny = self.bbox.min[1]
        mnz = self.bbox.min[2]
        
        mxx = self.bbox.max[0]
        mxy = self.bbox.max[1]
        mxz = self.bbox.max[2]
        
        trsf_mat = geomie3d.calculate.rotate_matrice((1,0,0), 0)
        
        if zup == False:
            trsf_mat = geomie3d.calculate.rotate_matrice((1,0,0), -90)
 
        geo_loc = self.geo_loc
        geo_loc_xyz = epsg4326_4978.transform(geo_loc[0], geo_loc[1], geo_loc[2])
        trsl_mat = geomie3d.calculate.translate_matrice(geo_loc_xyz[0], geo_loc_xyz[1], geo_loc_xyz[2])
        trsf_mat = trsf_mat@trsl_mat
        
        rad_list = []
        rad = True
        y_half_len = (mxy - mny)/2
        midy = mny + y_half_len
        west = (mnx, midy, mnz)
        west_mx = (mnx, midy, mxz)
        west = self.trsf_pos([west], trsf_mat)[0]
        west_mx = self.trsf_pos([west_mx], trsf_mat)[0]
        west_rad = epsg4978_4979.transform(west[0], west[1], west[2], radians = rad)
        west_rad_mx = epsg4978_4979.transform(west[0], west[1], west_mx[2], radians = rad)
        rad_list.append(list(west_rad))
        rad_list.append(list(west_rad_mx))
        
        x_half_len = (mxx - mnx)/2
        midx = mnx + x_half_len
        south = (midx, mny, mnz)
        south_mx = (midx, mny, mxz)
        south = self.trsf_pos([south], trsf_mat)[0]
        south_mx = self.trsf_pos([south_mx], trsf_mat)[0]
        south_rad = epsg4978_4979.transform(south[0], south[1], south[2], radians = rad)
        south_rad_mx = epsg4978_4979.transform(south[0], south[1], south_mx[2], radians = rad)
        rad_list.append(list(south_rad))
        rad_list.append(list(south_rad_mx))
        
        east = (mxx, midy, mnz)
        east_mx = (mxx, midy, mxz)
        east = self.trsf_pos([east], trsf_mat)[0]
        east_mx = self.trsf_pos([east_mx], trsf_mat)[0]
        east_rad = epsg4978_4979.transform(east[0], east[1], east[2], radians = rad)
        east_rad_mx = epsg4978_4979.transform(east[0], east[1], east_mx[2], radians = rad)
        rad_list.append(list(east_rad))
        rad_list.append(list(east_rad_mx))
        
        north = (midx, mxy, mnz)
        north_mx = (midx, mxy, mxz)
        north = self.trsf_pos([north], trsf_mat)[0]
        north_mx = self.trsf_pos([north_mx], trsf_mat)[0]
        
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
        self.bvol = {'region':region}
    
    def trsf_pos(self, pos_list, trsf_mat):
        if type(pos_list) != np.ndarray:
            np.array(pos_list)
        
        #add an extra column to the points
        npos = len(pos_list)
        xyzw = np.ones((npos,4))
        xyzw[:,:-1] = pos_list
        t_xyzw = np.dot(xyzw, trsf_mat.T)
        trsf_xyzs = t_xyzw[:,:-1]
        
        return trsf_xyzs.tolist()
    
    def compute_error(self):
        """
        This function only works if bbox is already computed
        """
        bbox = self.bbox
        lwr_left = [bbox.min[0], bbox.min[1], bbox.min[2]]
        upr_right =  [bbox.max[0], bbox.max[1], bbox.min[2]]
        dist = geomie3d.calculate.dist_btw_xyzs(lwr_left, upr_right)
        self.error = dist
    
    def add_box(self, box):
        """
        Parameters
        ----------
        bvol : Tuple
            Tuple specifying either box (https://github.com/CesiumGS/3d-tiles/tree/master/specification#bounding-volumes)
            e.g. bbox [0,0,0,
                       100,0,0,
                       0,100,0,
                       0,0,10]
        """
        self.bvol = {'box': box}
        
    
    def compute_enu2fixframe(self):
        """
        Compute the East North Up (ENU) coordinate to ECEF 4x4 transformation
        https://gis.stackexchange.com/questions/308445/local-enu-point-of-interest-to-ecef
        """
        lla_origin = self.geo_loc
        epsg4326_4978 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy = True)
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
        
        trsf_mat_flat = trsf_mat.T.flatten().tolist()
        self.transform = trsf_mat_flat
    
    def add_transform(self, transform):
        """
        Parameters
        ----------
        transform : Array
            1 dimension array of 8 elements. column major matrix e.g. [1, 0, 0, 0, 
                                                                       0, 1, 0, 0, 
                                                                       0, 0, 1, 0, 
                                                                       0, 0, 0, 1]
        """
        self.transform = transform
            
    def add_error(self, error):
        """
        Parameters
        ----------
        error : Float
            Geometric error of the node
        """
        self.error = error
        
    def edit_refine(self, refine_option):
        """
        Parameters
        ----------
        refine_option : str
            Options include "ADD", "REPLACE"
        """
        self.refine = refine_option
        
    def add(self, node):
        self.children.append(node)
    
    def all_nodes(self):
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes
        
    def to_dict(self, tile_path):
        node_dict = {}
        node_dict['boundingVolume'] = self.bvol
        node_dict['geometricError'] = self.error
        
        if self.transform != None:
            node_dict['transform'] = self.transform
        if self.tile_content != None:
            node_dict['content'] = {'uri': self.write_content(tile_path)}
        
        if len(self.children) > 0:    
            node_dict['refine'] = self.refine
            node_dict['children'] = [c.to_dict(tile_path) for c in self.children]

        return node_dict
        
    def write_content(self, tile_path):
        foldername = os.path.split(tile_path)[-1]
        content = self.tile_content
        magic = content.header.magic_value
        if magic == b"b3dm":
            filename = str(self.uniqid)+'.b3dm'
            file_path = os.path.join(tile_path, filename)
            f = open(file_path, 'wb')
            f.write(content.to_array())
            f.close()
            
            gltf = self.tile_content.body.glTF
            gltf_path = os.path.join(tile_path, str(self.uniqid)+'.glb')
            gltf.save(gltf_path)
    
            return foldername + '/' + filename