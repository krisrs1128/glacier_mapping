"""
Generate Masks from Tiffs / Shapefiles
"""
from joblib import Parallel, delayed
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
import argparse
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pathlib
import rasterio
import warnings
import yaml

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate_masks(img_paths, shps_paths, output_base="mask", out_dir=None, n_jobs=4):
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
        data_dir = os.environ["DATA_DIR"]
        out_dir = pathlib.Path(data_dir, "processed", "masks")

    pathlib.Path(out_dir).mkdir(parents=True)
    cols = ["id", "img", "mask", "img_width", "img_height", "mask_width", "mask_height"]
    metadata = pd.DataFrame({k: [] for k in cols})
    metadata_path = pathlib.Path(out_dir, "mask_metadata.csv")
    if not metadata_path.exists():
        metadata.to_csv(metadata_path, index=False)
    else:
        raise ValueError(f"Cannot overwrite {metadata_path}.")

    def wrapper(k):
        print(f"working on image {k + 1} / {len(img_paths)}")
        img, shps = rasterio.open(img_paths[k]), []
        for path in shps_paths[k]:
            gdf = gpd.read_file(path)
            gdf_crs = rasterio.crs.CRS.from_string(gdf.crs.to_string())
            if gdf_crs != img.meta["crs"]:
                gdf = gdf.to_crs(img.meta["crs"].data)
            shps.append(gdf)

        # build mask over tiff's extent, and save
        shps = clip_shapefile(img.bounds, img.meta, shps)
        mask = generate_mask(img.meta, shps)
        out_path = pathlib.Path(out_dir, f"{output_base}_{k:02}")
        np.save(str(out_path), mask)
        pd.DataFrame(
            {
                "img_path": img_paths[k],
                "mask": str(out_path) + ".npy",
                "width": img.meta["width"],
                "height": img.meta["height"],
                "mask_width": mask.shape[1],
                "mask_height": mask.shape[0],
            },
            index=[k],
        ).to_csv(metadata_path, header=False, mode="a")

    para = Parallel(n_jobs=n_jobs)
    para(delayed(wrapper)(k) for k in range(len(img_paths)))


def check_crs(a, b):
    if rasterio.crs.CRS.from_string(a.to_string()) != rasterio.crs.CRS.from_string(
        b.to_string()
    ):
        raise ValueError("Coordinate reference systems do not agree")


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
    result = np.zeros((img_meta["height"], img_meta["width"], len(shps)), dtype="uint8")
    for k, shp in enumerate(shps):
        check_crs(img_meta["crs"], shp.crs)
        result[:, :, k] = channel_mask(img_meta, shp)
    result[:, :, 0] = np.multiply(result[:, :, 0], result[:, :, 1])
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

        else:  # case if multipolygon
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
        poly_pts.append(~transform * tuple(i)[:2])  # in case polygonz format
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
    bbox_poly = gpd.GeoDataFrame(
        {"geometry": bbox}, index=[0], crs=img_meta["crs"].data
    )
    result = []
    for shp in shps:
        check_crs(img_meta["crs"], shp.crs)
        result.append(shp.loc[shp.intersects(bbox_poly["geometry"][0])])
    return result


if __name__ == "__main__":
    root_dir = pathlib.Path(os.environ["ROOT_DIR"])
    masking_conf = root_dir / "conf" / "masking_paths.yaml"

    parser = argparse.ArgumentParser(
        description="Defining label masks from tiff + shapefile pairs"
    )
    parser.add_argument(
        "-m",
        "--masking_conf",
        default=masking_conf,
        help="yaml file specifying which shapefiles to burn onto tiffs. See conf/masking_paths.yaml for an example.",
    )
    args = parser.parse_args()

    masking_paths = yaml.safe_load(open(args.masking_conf, "r"))
    img_paths = [p["img_path"] for p in masking_paths.values()]
    mask_paths = [p["mask_paths"] for p in masking_paths.values()]
    generate_masks(img_paths, mask_paths, n_jobs=1)
