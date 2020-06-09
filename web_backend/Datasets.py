#!/usr/bin/env python3
import shapely
import shapely.geometry
import web_backend.DataLoader as DL
import fiona
import fiona.transform
import json
import os
import utm

_DATASET_FN = "conf/dataset.json"
DATA_DIR = os.environ["DATA_DIR"]

def get_area_from_geometry(geom, src_crs="epsg:4326"):
    if geom["type"] == "Polygon":
        lon, lat = geom["coordinates"][0][0]
    elif geom["type"] == "MultiPolygon":
        lon, lat = geom["coordinates"][0][0][0]
    else:
        raise ValueError("Polygons and MultiPolygons only")

    zone_number = utm.latlon_to_zone_number(lat, lon)
    hemisphere = "+north" if lat > 0 else "+south"
    dest_crs = "+proj=utm +zone=%d %s +datum=WGS84 +units=m +no_defs" % (zone_number, hemisphere)
    projected_geom = fiona.transform.transform_geom(src_crs, dest_crs, geom)
    area = shapely.geometry.shape(projected_geom).area / 1000000.0 # we calculate the area in square meters then convert to square kilometers
    return area

def _load_geojson_as_list(fn):
    ''' Takes a geojson file as input and outputs a list of shapely `shape` objects in that file and their corresponding areas in km^2.

    We calculate area here by re-projecting the shape into its local UTM zone, converting it to a shapely `shape`, then using the `.area` property.
    '''
    shapes = []
    areas = []
    crs = None
    with fiona.open(fn) as f:
        src_crs = f.crs
        for row in f:
            geom = row["geometry"]

            area = get_area_from_geometry(geom, src_crs)
            areas.append(area)

            shape = shapely.geometry.shape(geom)
            shapes.append(shape)
    return shapes, areas, src_crs


def _load_dataset(dataset):
    # Step 2: make sure the dataLayer exists
    if dataset["dataLayer"]["type"] == "CUSTOM":
        fn = os.path.join(DATA_DIR, dataset["dataLayer"]["path"])
        if not os.path.exists(fn):
            return False # TODO: maybe we should make these errors more descriptive (explain why we can't load a dataset)

    # Step 3: setup the appropriate DatasetLoader
    if dataset["dataLayer"]["type"] == "GLACIER":
        data_loader = DL.DataLoaderGlacier(dataset["dataLayer"]["padding"], dataset["dataLayer"]["path"])
    else:
        raise ValueError(f"Cannot find loader for {dataset['dataLayer']['type']}")

    return {
        "data_loader": data_loader,
    }

def load_dataset():
    dataset_json = json.load(open(os.path.join(os.environ["ROOT_DIR"], _DATASET_FN),"r"))
    return _load_dataset(dataset_json)
