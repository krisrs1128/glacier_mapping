"""
Generate Masks from Tiffs / Shapefiles
"""
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
import geopandas as gpd
import numpy as np
import pathlib
import rasterio

def generate_masks(img_metas, matching_shps, output_base="mask", out_dir=None):
    """
    A wrapper of generate_mask, to make labels for each input
    """
    masks = []
    for k, img_meta in enumerate(img_metas):
        mask = generate_mask(img_meta, matching_shps[k])
        if outdir is None:
            masks.append(mask)
        else:
            # convert numpy to tiff
            # save the tiff

    return masks


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


if __name__ == '__main__':
    img_dir = "/scratch/sankarak/data/glaciers_azure/img_data/2010/nepal/"
    img_ids = ["Nepal_139041_20111225.tif", "Nepal_140041_20091108.tif", "Nepal_141040_20101204.tif"]
    img_paths = [pathlib.Path(img_dir, s) for s in img_ids]
    img_metas = [rasterio.open(p).meta for p in img_paths]
    shps = convert_crs(img.meta, [gpd.read_file(labels_path), gpd.read_file(borders_path)])
    masks = generate_masks(img_metas, len(img_metas) * [shps])
