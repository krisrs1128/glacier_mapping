"""
Generate Masks from Tiffs / Shapefiles
"""
from joblib import Parallel, delayed
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pathlib
import rasterio
import re
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate_masks(img_paths, shps_paths, output_base="mask",
                   out_dir=None, n_jobs=4):
    """
    A wrapper of generate_mask, to make labels for each input

    :param img_meta: The metadata field associated with a geotiff. Expected to
      contain transform (coordinate system), height, and width fields.
    :param img_bounds: A list of coordinates specifying the bounding box of the
      input img.
    :param shps_paths: A list of lists of paths to shapefiles. The k^th element
      is a list of paths, each of which will become a channel in the k^th
      resulting mask.
    :param output_base: The basename for all the output numpy files
    :param out_dir: The directory to which to save all the results.
    """
    if not out_dir:
        out_dir = os.getcwd()

    cols = ["id", "img", "mask", "img_width", "img_height", "mask_width", "mask_height"]
    metadata = pd.DataFrame({k: [] for k in cols})
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    metadata_path = pathlib.Path(out_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    def wrapper(k):
        print(f"working on image {k} / {len(img_paths)}")
        img, shps = rasterio.open(img_paths[k]), []
        
        # print(shps_paths)
        for path in shps_paths[k]:
            gdf = gpd.read_file(path)
            gdf_crs = rasterio.crs.CRS(gdf.crs)
            if gdf_crs != img.meta["crs"]:
                gdf = gdf.to_crs(img.meta["crs"].data)
            shps.append(gdf)

        if rasterio.crs.CRS(img.meta["crs"].data) != (rasterio.crs.CRS(shps[0].crs) == rasterio.crs.CRS(shps[1].crs)):
            print("\nImageCRS: ",rasterio.crs.CRS(img.meta["crs"].data)) 
            print("\nGlaciersCRS: ",rasterio.crs.CRS(shps[0].crs))
            print("\nBordersCRS: ",rasterio.crs.CRS(shps[1].crs))

        # build mask over tiff's extent, and save
        shps = clip_shapefile(img.bounds, img.meta, shps)
        mask = generate_mask(img.meta, shps)
        out_path = pathlib.Path(out_dir, f"{output_base}_{k:02}")
        np.save(str(out_path), mask)
        pd.DataFrame({
            "img_path": img_paths[k],
            "mask": str(out_path) + ".npy",
            "width": img.meta["width"],
            "height": img.meta["height"],
            "mask_width": mask.shape[1],
            "mask_height": mask.shape[0]
        }, index=[k]).to_csv(metadata_path, header = False, mode="a")

    para = Parallel(n_jobs=n_jobs)
    para(delayed(wrapper)(k) for k in range(len(img_paths)))


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
    result = np.zeros((img_meta["height"], img_meta["width"], len(shps)))
    for k, shp in enumerate(shps):
        if rasterio.crs.CRS(img_meta["crs"]) != rasterio.crs.CRS(shp.crs):
            raise ValueError("Coordinate reference systems do not agree")
        result[:, :, k] = channel_mask(img_meta, shp)
    result[:,:,0] = np.multiply(result[:,:,0], result[:,:,1])
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

    im_size = (img_meta["height"], img_meta["width"])
    return rasterize(shapes=poly_shp, out_shape=im_size)


def poly_from_coord(polygon, transform):
    """
    Get a transformed polygon
    https://lpsmlgeo.github.io/2019-09-22-binary_mask/
    """
    poly_pts = []
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i)[:2]) # in case polygonz format
    return Polygon(poly_pts)


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
        if rasterio.crs.CRS(img_meta["crs"]) != rasterio.crs.CRS(shp.crs):
            raise ValueError("Coordinate reference systems do not agree")

        result.append(shp.loc[shp.intersects(bbox_poly["geometry"][0])])
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
    for k, img_path in enumerate(img_paths):
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


if __name__ == "__main__":
    img_dir = "/scratch/sankarak/data/glaciers_azure/"
    paths_df = path_pairs_landsat(img_dir)
    img_paths = paths_df["img"].values
    shps_paths = [[p["label"], p["border"]] for _, p in paths_df.iterrows()]
    out_dir = "/scratch/sankarak/data/glaciers_azure/masks"
    generate_masks(img_paths, shps_paths, out_dir=out_dir, n_jobs=1)
    paths_df.to_csv(out_dir + "/paths.csv", index=False)
