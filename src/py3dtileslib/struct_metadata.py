from pygltflib import GLTF2

def add_prop_table2featureid(prop_table_id: int, featureid: dict):
    """
    add the index of the property table to the featureid of a primitive with the EXT_mesh_features
    """
    featureid['propertyTable'] = prop_table_id

def add_extstructmetadata(gltf: GLTF2, schema_id: str, classes: dict, prop_tables: list[dict], enums: dict = None):
    """
    takes in a gltf object and add the EXT_structural_metadata extension into the gltf object

    Parameters
    ----------
    enums : dict
        dictionary of enum. Generated from the function add_enum2enums. It should be in this format 
        {'example_enum_table': {'name': table_name, 'description': 'this is an example', 'values': [{'name': 'x', 'value':0}, {'name': 'y', 'value':1}]}}

    """
    ext = gltf.extensions
    ext_used = gltf.extensionsUsed

    ext['EXT_structural_metadata'] = {'schema': {'id': schema_id, 'classes': {classes['name']: classes}}}
    ext['EXT_structural_metadata']['propertyTables'] = prop_tables
    if enums is not None:
        ext['EXT_structural_metadata']['schema']['enums'] = enums

    if 'EXT_structural_metadata' not in ext_used:
        ext_used.append('EXT_structural_metadata')

    gltf.extensions = ext
    gltf.extensionsUsed = ext_used

def add_property2classes(classes: dict, property_id: str, property: dict):
    """
    add properties to the classes json
    """
    classes['properties'][property_id] = property

def create_classes( name: str, description: str) -> dict:
    """
    create a class for the EXT_structural_metadata
    """
    classes_json = {'name': name, 'description': description, 'properties': {}}
    return classes_json

def create_classes_prop(name: str, description: str, prop_type: str, comptype: str = None, 
                        enumtype: str = None) -> dict:
    """
    create a class property. 
    refer to https://github.com/CesiumGS/glTF/blob/3d-tiles-next/extensions/2.0/Vendor/EXT_structural_metadata/schema/class.property.schema.json to understand the schema 
    Refer to https://github.com/CesiumGS/3d-tiles/tree/main/specification/Metadata#property for understanding the different types.

    Parameters
    ----------
    prop_type : str
        'SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4', 'STRING', 'BOOLEAN', 'ENUM'.

    comptype : str, optional
        If is 'STRING', this will be None. 'INT8', 'UINT8', 'INT16', 'UINT16', 'INT32', 'UINT32', 'INT64', 'UINT64', 'FLOAT32', 'FLOAT64'.
    
    enumtype : str, optional
        Enum ID as declared in the `enums` dictionary. Required when prop type is `ENUM`.
    """
    class_prop = {'name': name, 'description': description, 'type': prop_type}
    if comptype is not None:
        class_prop['componentType'] = comptype 
    if enumtype is not None:
        class_prop['enumType'] = enumtype 
    return class_prop

def create_prop_table(name: str, class_name: str, count: int) -> dict:
    """
    create property table

    Parameters
    ----------
    class_name : str
        The class that property values conform to. The value must be a class ID declared in the `classes` dictionary.

    count : int
        Number of elements in each property. If you have 4 feature id, count is 4.
    """
    prop_table_json = {'name': name, 'class': class_name, 'count': count, 'properties': {}}
    return prop_table_json

def add_table_property(prop_table: dict, prop_id: str, values: int, string_offset: int = None, arr_offset: int = None) -> dict:
    """
    add property to a table. The column of each table.

    Parameters
    ----------
    prop_id : str
        corresponds to a property ID in the class' `properties` dictionary 
        
    values : int
        the index of the bufferview.
    
    string_offset : int, optional
        the index of the bufferview for string offset.
    
    arr_offset : int, optional
        the index of the bufferview for array offset.
    """
    prop_table['properties'][prop_id] = {'values': values}
    if string_offset is not None:
        prop_table['properties'][prop_id]['stringOffsets'] = string_offset
    if arr_offset is not None:
        prop_table['properties'][prop_id]['arrayOffsets'] = arr_offset

def create_enum(name: str, description: str, values: list[dict]) -> dict:
    """
    create an enum json

    Parameters
    ----------
    values : list[dict]
        the enumerations in this format: [{'name': 'x', 'value':0}, {'name': 'y', 'value':1}]

    """
    return {'name': name, 'description': description, 'values': values}

def add_enum2enums(enums: dict, enum_id: str, enum: dict):
    """
    update the enums dictionary with the enum dictionary with the id being the key
    """
    enums[enum_id] = enum