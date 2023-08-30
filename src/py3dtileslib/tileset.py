import os
import json

from pygltflib import GLTF2

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
    def __init__(self, folder_path: str, version: float) -> None:
        self.properties = None
        self.error = None
        self.root = None
        self.asset = {'version':str(version)}
        self.folder_path = folder_path
    
    def add_root(self, node):
        """
        Parameters
        ----------
        node : Node
            Node object
        """
        self.root = node
        
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
        self.transform = None
        self.bvol = None
        self.error = None
        self.refine = 'ADD' #other options include "REPLACE"
        self.tile_content = None #TileContent object B3DM PNT I3DM GLTF
        self.children = []
        self.uniqid = uniqid
        
    def add_tile_content(self, tile_content):
        """
        Parameters
        ----------
        tile_content : TileContent object
            TileContent object B3DM PNT I3DM GLTF
        """
        self.tile_content = tile_content
    
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
        
    def add_bregion(self, region):
        """
        Parameters
        ----------
        region : Tuple
            Tuple specifying either box (https://github.com/CesiumGS/3d-tiles/tree/master/specification#bounding-volumes)
            e.g. bbox [0,0,0,
                       100,0,0,
                       0,100,0,
                       0,0,10]
        """
        self.bvol = {'region': region}
    
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
        else:
            node_dict['refine'] = self.refine

        return node_dict
        
    def write_content(self, tile_path):
        foldername = os.path.split(tile_path)[-1]
        content = self.tile_content
        if type(content) == GLTF2:
            filename = str(self.uniqid)+'.gltf'
            file_path = os.path.join(tile_path, filename)
            content.save(file_path)
            return foldername + '/' + filename
        else:
            magic = content.header.magic_value
            if magic == b"b3dm":
                filename = str(self.uniqid)+'.b3dm'
                file_path = os.path.join(tile_path, filename)
                content.save_as(file_path)
                # region: for debugging
                # gltf = self.tile_content.body.glTF
                # gltf_path = os.path.join(tile_path, str(self.uniqid)+'.glb')
                # gltf.save(gltf_path)
                # endregion
                return foldername + '/' + filename
            
            elif magic == b"pnts":
                filename = str(self.uniqid)+'.pnts'
                file_path = os.path.join(tile_path, filename)
                content.save_as(file_path)
                return foldername + '/' + filename
            
            elif magic == b"i3dm":
                filename = str(self.uniqid)+'.i3dm'
                file_path = os.path.join(tile_path, filename)
                content.save_as(file_path)
                return foldername + '/' + filename