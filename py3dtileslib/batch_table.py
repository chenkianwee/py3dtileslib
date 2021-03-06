import struct
import numpy as np
import json


class BatchTable(object):
    def __init__(self):
        self.header = BatchTableHeader()
        self.body = BatchTableBody()

    def add_property(self, property_name, array):
        """
        Parameters
        ----------
        property_name : 'str'
            name of the property
        
        array :  list
            array of the properties

        """
        
        self.header.add_property(property_name, array)
        
    def add_binary_property(self, property_name, component_type, batch_type, array):
        """
        Parameters
        ----------
        property_name : 'str'
            name of the property

        component_type : str
             "BYTE", "UNSIGNED_BYTE", "SHORT", "UNSIGNED_SHORT","INT","UNSIGNED_INT","FLOAT","DOUBLE"
        
        batch_type : 'str'
            'SCALAR', 'VEC2', 'VEC3', 'VEC4'
        
        array :  ndarray
            array of the properties

        """
        header = self.header
        self.body.add_property(header, property_name, component_type, batch_type, array)
    
    def add_batch_length(self, batch_length):
        self.header.add_batch_length(batch_length)
        
    # returns batch table as binary
    def to_array(self, tile_type, ft_len):
        header_arr = self.header.to_array(tile_type, ft_len)
        bth_len = len(header_arr)
        body_arr = self.body.to_array(self.header, tile_type, ft_len, bth_len)
        if len(body_arr) == 0:
            return header_arr
        else:
            bt_arr =  np.concatenate((header_arr, body_arr))
            return bt_arr
    
    @staticmethod
    def from_array(th, ft, array):
        """
        Parameters
        ----------
        th : TileHeader

        array : numpy.array

        Returns
        -------
        bt : BatchTable
        """
        # build feature table
        bt = BatchTable()
       
        # build batch table header
        bth_len = th.bt_json_byte_length
        bth_arr = array[0:bth_len]
        nbatch = None
        if ft.header.batch_length !=None:
            nbatch = ft.header.batch_length 
        elif ft.header.points_length != None:
            nbatch = ft.header.points_length
        elif ft.header.instances_length != None:
            nbatch = ft.header.instances_length
        
        bth = BatchTableHeader.from_array(bth_arr, nbatch)
        bt.header = bth
         
        # build batch table body
        btb_len = th.bt_bin_byte_length
      
        btb_arr = array[bth_len:bth_len + btb_len]
        btb = BatchTableBody.from_array(bth, btb_arr)
        bt.body = btb
        
        return bt       
        
class BatchTableHeader(object):
    def __init__(self):
        self.properties = {}
        self.property_names = []
        self.batch_length = None

    def add_property(self, propertyName, array):
        if type(array) == np.ndarray:
            array = array.tolist()
            
        self.properties[propertyName] = array
        self.property_names.append(propertyName)
    
    def add_batch_length(self, batch_length):
         self.batch_length = batch_length
        
    # returns batch table as binary
    def to_array(self, tile_type, ft_len):
        # convert dict to json string
        bt_json = json.dumps(self.properties, separators=(',', ':'))
        # header must be 4-byte aligned (refer to batch table documentation)
        if tile_type == 'b3dm' or tile_type == 'pnts':
            n = 28 + ft_len + len(bt_json)
        elif tile_type == 'i3dm':
            n = 32 +ft_len + len(bt_json)
        
        if n%8 !=0:
            bt_json += ' ' * (8 - n % 8)
        # returns an array of binaries representing the batch table
        return np.fromstring(bt_json, dtype=np.uint8)
    
    @staticmethod
    def from_array(array, batch_length):
        """
        Parameters
        ----------
        array : numpy.array

        """
        bth = BatchTableHeader()
        bth.batch_length = batch_length
        bt_json_str = ''.join([c.decode('UTF-8') for c in array.view('c')])
        bt_json = json.loads(bt_json_str)
        
        bth.properties = bt_json
        bth.property_names = list(bt_json.keys())
        return bth
        
class BatchTableBody(object):
    def __init__(self):
        self.property_arr = {}

    def add_property(self, bth, property_name, component_type, batch_type, array):
        """
        Parameters
        ----------
        bth: BatchTableHeader()
            batchtable header object    
        
        property_name : 'str'
            name of the property

        component_type : str
             "BYTE", "UNSIGNED_BYTE", "SHORT", "UNSIGNED_SHORT","INT","UNSIGNED_INT","FLOAT","DOUBLE"
        
        batch_type : 'str'
            'SCALAR', 'VEC2', 'VEC3', 'VEC4'
        
        array :  ndarray
            array of the properties

        """
        if type(array) == np.ndarray:
            array = array.tolist()
        
        nbatch = bth.batch_length
        narr = len(array)
        if nbatch != narr:
            raise Exception("number of array does not match batch length")
            
        #figure out the offset
        property_names = bth.property_names
        prop_arr = self.property_arr
        keys = list(prop_arr.keys())
        
        offset = 0
        for name in property_names:
            if name in keys:
                bcnt = len(prop_arr[name])
                offset = offset + bcnt
        
        bth.property_names.append(property_name)
        bth.properties[property_name] = {'byteOffset' : offset, 'componentType' : component_type, 'type': batch_type}
            
        if component_type == "BYTE":
            com_type = np.byte
            # prop_dt = create_dt(dt_names, np.byte)
        elif component_type == "UNSIGNED_BYTE":
            com_type = np.ubyte
        elif component_type == "SHORT":
            com_type = np.short
        elif component_type == "UNSIGNED_SHORT":
            com_type = np.ushort
        elif component_type == "INT":
            com_type = np.intc
        elif component_type == "UNSIGNED_INT":
            com_type = np.uintc
        elif component_type == "FLOAT":
            com_type = np.float32
        elif component_type == "DOUBLE":
            com_type = np.double
        
        prop_arr = np.array(array, dtype=com_type).view('uint8')
        self.property_arr[property_name] = prop_arr

    # returns batch table as binary
    def to_array(self, bth, tile_type, ft_len, bth_len):
        prop_arr = self.property_arr
        property_names = bth.property_names
        
        keys = list(prop_arr.keys())
        arr = np.array([])
        for name in property_names:
            if name in keys:
                if len(arr) == 0 :
                    arr = prop_arr[name]
                else:
                    arr = np.concatenate((arr, prop_arr[name]))
        
        if len(arr) !=0:
            if tile_type == 'pnts' or tile_type == 'b3dm': 
                n = 28 + ft_len + bth_len + len(arr)
            elif tile_type == 'i3dm':
                n = 32 + ft_len + bth_len + len(arr)
            
            if n%8 !=0:
                add_byte = 8 - n%8
                add_arr = np.array([0], dtype = np.byte).view('uint8')
                add_arr = np.repeat(add_arr, add_byte)
                arr = np.concatenate((arr, add_arr))
        
        return arr
    
    @staticmethod
    def from_array(bth, array):
        """
        Parameters
        ----------
        array : numpy.array

        """
        btb = BatchTableBody()
        
        if len(array) != 0:
            prop = bth.properties
            for k in prop.keys():
                val = prop[k]
                if type(val) == dict:
                    val_keys = val.keys()
                    if 'byteOffset' in val_keys and 'componentType' in val_keys and 'type' in val_keys:
                        #this is bt binary property
                        offset = val['byteOffset']
                        com_type = val['componentType']
                        prop_itemsize = 0
                        if com_type == "BYTE":
                            prop_itemsize = 1
                        elif com_type == "UNSIGNED_BYTE":
                            prop_itemsize = 1
                        elif com_type == "SHORT":
                            prop_itemsize = 2
                        elif com_type == "UNSIGNED_SHORT":
                            prop_itemsize = 2
                        elif com_type == "INT":
                            prop_itemsize = 4
                        elif com_type == "UNSIGNED_INT":
                            prop_itemsize = 4
                        elif com_type == "FLOAT":
                            prop_itemsize = 4
                        elif com_type == "DOUBLE":
                            prop_itemsize = 8
                        
                        batchtype = val['type']
                        if batchtype == 'SCALAR':
                            prop_itemsize = prop_itemsize*1
                        elif batchtype == 'VEC2':
                            prop_itemsize = prop_itemsize*2
                        elif batchtype == 'VEC3':
                            prop_itemsize = prop_itemsize*3
                        elif batchtype == 'VEC4':
                            prop_itemsize = prop_itemsize*4
                        
                        nbatch = bth.batch_length
                        prop_arr = array[offset:offset + nbatch * prop_itemsize]
                        btb.property_arr[k] = {'arr': prop_arr}
        return btb
    
    def unpack_properties(self, bth):
        """
        unpack the properties into their respective types for readability

        """
        bth_prop = bth.properties
        prop_dict = self.property_arr
        keys = prop_dict.keys()
        unpack_dict = {}
        for key in keys:
            meta = bth_prop[key]
            com_type = meta['componentType']
            format_str = ''
            prop_itemsize = 0
            if com_type == "BYTE":
                format_str = 's'
                prop_itemsize = 1
            elif com_type == "UNSIGNED_BYTE":
                format_str = 's'
                prop_itemsize = 1
            elif com_type == "SHORT":
                format_str = 'h'
                prop_itemsize = 2
            elif com_type == "UNSIGNED_SHORT":
                format_str = 'H'
                prop_itemsize = 2
            elif com_type == "INT":
                format_str = 'i'
                prop_itemsize = 4
            elif com_type == "UNSIGNED_INT":
                format_str = 'I'
                prop_itemsize = 4
            elif com_type == "FLOAT":
                format_str = 'f'
                prop_itemsize = 4
            elif com_type == "DOUBLE":
                format_str = 'd'
                prop_itemsize = 8
                
            batchtype = meta['type']
            if batchtype == 'SCALAR':
                format_str = format_str
                prop_itemsize = prop_itemsize*1
            elif batchtype == 'VEC2':
                format_str = format_str*2
                prop_itemsize = prop_itemsize*2
            elif batchtype == 'VEC3':
                format_str = format_str*3
                prop_itemsize = prop_itemsize*3
            elif batchtype == 'VEC4':
                format_str = format_str*4
                prop_itemsize = prop_itemsize*4
            
            arr = prop_dict[key]
            nbatch = bth.batch_length
            
            unpack_arr = []
            for i in range(nbatch):
                index = i*prop_itemsize
                d = arr[index:index+prop_itemsize]
                val = struct.unpack(format_str, d)
                if len(val) == 1:
                    unpack_arr.append(val[0])
                else:
                     unpack_arr.append(val)
                     
            unpack_dict[key] = unpack_arr
            
        return unpack_dict