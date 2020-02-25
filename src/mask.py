"""
Generate Masks from Tiffs / Shapefiles
"""
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
import skimage.io
import glob
import geopandas as gpd
import numpy as np
import pathlib
import rasterio
import pandas as pd
import os

def generate_masks(img_metas, shps_paths, output_base="mask", out_dir=None):
    """
    A wrapper of generate_mask, to make labels for each input
    """
    if not out_dir:
        out_dir = os.getcwd()

    shape_objects = {}
    for k, img_meta in enumerate(img_metas):
        # if current shape paths are in objects, use them to make mask
        # otherwise read in, add to object, and use for mask
        print(f"working on {img_meta} ({k} / {len(img_metas)})")

        shps = []
        for path in shps_paths[k]:
            if path not in shape_objects.keys():
                shape_objects[path] = gpd.read_file(path)
                shape_objects[path] = shape_objects[path].to_crs(img_meta["crs"].data)

            shps.append(shape_objects[path])

        mask = generate_mask(img_meta, shps)
        out_path = pathlib.Path(out_dir, f"{output_base}_{k}.tiff")
        # skimage.io.imsave(str(out_path), mask, plugin="tifffile")
        np.save(str(out_path), mask)


# def generate_masks(img_metas, matching_shps, output_base="mask", out_dir=None):
#     """
#     A wrapper of generate_mask, to make labels for each input
#     """
#     masks = []
#     for k, img_meta in enumerate(img_metas):
#         mask = generate_mask(img_meta, matching_shps[k])
#         if outdir is None:
#             masks.append(mask)
#         else:
#             # convert numpy to tiff
#             # save the tiff
#             pass

#     return masks

def generate_mask(img_meta, shps):
    """
    Generate K-Channel Label Masks over Raster Image

    :param img_meta: The metadata field associated with a geotiff. Expected to
      contain transform (coordinate system), height, and width fields.
    :param shps: A list of K geopandas shapefiles, used to build the mask.
      Assumed to be in the same coordinate system as img_data.
    :return mask: A K channel binary numpy array. The k^th channel gives the
      binary mask for the k^th input shapefile.
    """
    result = np.zeros((img_meta["width"], img_meta["height"], len(shps)))
    for k, shp in enumerate(shps):
        if img_meta["crs"].to_string() != shp.crs.to_string():
            raise ValueError("Coordinate reference systems do not agree")
        result[:, :, k] = channel_mask(img_meta, shp)

    return result


def channel_mask(img_meta, shp):
    """
    Generate 1-Channel Label Mask over Raster Image

    :param shp: A geopandas shapefile, used to build the mask.
    """
    poly_shp = []
    for _, row in shp.iterrows():
        if row["geometry"].geom_type == "Polygon":
            poly_shp += [poly_from_coord(row["geometry"], img_meta["transform"])]
        else: # case if multipolygon
            for p in row["geometry"]:
                poly_shp += [poly_from_coord(p, img_meta["transform"])]

    im_size = (img_meta["width"], img_meta["height"])
    return rasterize(shapes=poly_shp, out_shape=im_size)


#Generate polygon
def poly_from_coord(polygon, transform):
    """
    https://lpsmlgeo.github.io/2019-09-22-binary_mask/
    """
    poly_pts = []
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))

    # Generate a polygon object
    return Polygon(poly_pts)


def convert_crs(img_meta, shps):
    """
    Convert shapefile CRS to img CRS
    """
    result = []
    for shp in shps:
        result += [shp.to_crs(img_meta["crs"].data)]
    return result

def clip_shapefile(img_bounds, img_meta, shps):
    """
    Clip Shapefile Extents to Image Bounding Box

    :param img_bounds: The rectangular lat/long bounding box associated with a
      raster tiff.
    :param img_meta: The metadata field associated with a geotiff. Expected to
      contain transform (coordinate system), height, and width fields.
    :param shps: A list of K geopandas shapefiles, used to build the mask.
      Assumed to be in the same coordinate system as img_data.
    :return result: The same shapefiles as shps, but with polygons that don't
      overlap the img bounding box removed.
    """
    bbox = box(*img_bounds)
    bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=img_meta["crs"].data)

    result = []
    for shp in shps:
        if img_meta["crs"].to_string() != shp.crs.to_string():
            raise ValueError("Coordinate reference systems do not agree")

        result.append(gpd.overlay(shp, bbox_poly))
    return result

def parse_path(path):
    #
    # example path: /scratch/sankarak/data/glaciers_azure/img_data/2005/nepal
    # year is the number right after data/
    # https://regexr.com/4ponn
    regexes = re.compile("(data\/)([0-9]+)(\/)([A-z]+)").search(str(path))
    _, year, _, region = regexes.groups()
    return year, region

def path_pairs_landsat(base_dir):
    """
    Mapping from Tiffs to their Labels

    For the landsat 7 data whose IDs were given to us by ICIMOD's shapefiles,
    this gives the mapping between the raw tiff file and the corresponding
    shapefiles / borders.

    :param base_dir: The directory containing img_data and vector_data. See
      ee_codes/Readme.md for a description of the directory structure.
    :return pairs: A pandas dataframe with columns "img", "label", and
      "border". "img" gives the path to the raw tiffile, "label" is the path to
      the shapefile for that tiff, and "borders" is the path to the shapefile
      for the border of the enclosing country.
    """
    pairs = []
    img_paths = pathlib.Path(base_dir).glob("**/*tif")
    for img_path in img_paths:
        year, region = parse_path(img_path)
        label_dir = pathlib.Path(base_dir, "vector_data", year, region, "data")
        label_path = pathlib.Path(label_dir, f"Glacier_{year}.shp")

        border_path = pathlib.Path(base_dir, "vector_data", "borders", region, f"{region}.shp")
        pairs.append({
            "img": str(img_path),
            "label": str(label_path),
            "border": str(border_path)
        })

    return pd.DataFrame(pairs)


if __name__ == '__main__':
    img_dir = "/scratch/sankarak/data/glaciers_azure/"
    paths_df = path_pairs_landsat(img_dir)
    img_metas = [rasterio.open(p).meta for p in paths_df["img"].values]
    shps_paths =[[p["label"], p["border"]] for _, p in paths_df.iterrows()]
    generate_masks(img_metas, shps_paths)
