#!/usr/bin/env python3
from web_backend.DataLoaderAbstract import DataLoader
from pathlib import Path
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
import base64
import cv2
import fiona
import fiona.transform
import numpy as np
import os
import rasterio
import rasterio.crs
import rasterio.io
import rasterio.mask
import rasterio.warp
import shapely.geometry

# ------------------------------------------------------
# Miscellaneous methods
# ------------------------------------------------------
REPO_DIR = os.environ["REPO_DIR"]

def extent_to_transformed_geom(extent, dest_crs):
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    return fiona.transform.transform_geom(
        f"EPSG:{extent['crs']}",
        dest_crs,
        geom
    )


def warp_data(src_img, src_crs, src_transform, src_bounds, dest_epsg=3857, resolution=10):
    ''' Assume that src_img is (height, width, channels)
    '''
    assert len(src_img.shape) == 3
    src_height, src_width, num_channels = src_img.shape
    src_img_tmp = np.rollaxis(src_img.copy(), 2, 0)

    dst_crs = rasterio.crs.CRS.from_epsg(dest_epsg)
    dst_bounds = rasterio.warp.transform_bounds(src_crs, dst_crs, *src_bounds)
    dst_transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=src_width, height=src_height,
        left=src_bounds[0],
        bottom=src_bounds[1],
        right=src_bounds[2],
        top=src_bounds[3],
        resolution=resolution
    )

    with rasterio.io.MemoryFile() as memfile:
        dst_file = memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=src_img_tmp.shape[0],
            dtype=np.float32,
            crs=dst_crs,
            transform=dst_transform
        )
        for k in range(src_img_tmp.shape[0]):
            dst_file.write(src_img_tmp[k], k + 1)

    dst_img = dst_file.read()
    dst_img = np.transpose(dst_img, (1, 2, 0))
    return dst_img, dst_bounds


def encode_rgb(x):
    x_im = cv2.imencode(".png", cv2.cvtColor(x, cv2.COLOR_RGB2BGR))[1]
    return base64.b64encode(x_im.tostring()).decode("utf-8")


# ------------------------------------------------------
# DataLoader for arbitrary GeoTIFFs
# ------------------------------------------------------
class DataLoaderCustom(DataLoader):

    @property
    def shapes(self):
        return self._shapes
    @shapes.setter
    def shapes(self, value):
        self._shapes = value

    @property
    def padding(self):
        return self._padding
    @padding.setter
    def padding(self, value):
        self._padding = value

    def __init__(self, data_fn, shapes, padding):
        self.data_fn = data_fn
        self._shapes = shapes
        self._padding = padding

    def get_data_from_extent(self, extent):
        f = rasterio.open(Path(REPO_DIR, self.data_fn), "r")
        src_index = f.index
        src_crs = f.crs
        transformed_geom = extent_to_transformed_geom(extent, f.crs.to_dict())
        transformed_geom = shapely.geometry.shape(transformed_geom)
        buffed_geom = transformed_geom.buffer(self.padding)
        geom = shapely.geometry.mapping(shapely.geometry.box(*buffed_geom.bounds))

        # passed into the model
        src_image, src_transform = rasterio.mask.mask(f, [geom], crop=True)
        f.close()
        return src_image, src_crs, src_transform, buffed_geom.bounds, src_index

    def get_area_from_shape_by_extent(self, extent, shape_layer):
        i, shape = self.get_shape_by_extent(extent, shape_layer)
        return self.shapes[shape_layer]["areas"][i]

    def get_data_from_shape_by_extent(self, extent, shape_layer):
        # First, figure out which shape the extent is in
        _, shape = self.get_shape_by_extent(extent, shape_layer)
        mask_geom = shapely.geometry.mapping(shape)

        # Second, crop out that area for running the entire model on
        f = rasterio.open(Path(REPO_DIR, self.data_fn), "r")
        src_profile = f.profile
        src_crs = f.crs.to_string()
        src_bounds = f.bounds
        transformed_mask_geom = fiona.transform.transform_geom(self.shapes[shape_layer]["crs"], src_crs, mask_geom)
        src_image, src_transform = rasterio.mask.mask(f, [transformed_mask_geom], crop=True, all_touched=True, pad=False)
        f.close()

        return src_image, src_profile, src_transform, shapely.geometry.shape(transformed_mask_geom).bounds, src_crs

    def get_shape_by_extent(self, extent, shape_layer):
        transformed_geom = extent_to_transformed_geom(extent, self.shapes[shape_layer]["crs"])
        transformed_shape = shapely.geometry.shape(transformed_geom)
        mask_geom = None
        for i, shape in enumerate(self.shapes[shape_layer]["geoms"]):
            if shape.contains(transformed_shape.centroid):
                return i, shape
        raise ValueError("No shape contains the centroid")

    def get_data_from_shape(self, shape):
        mask_geom = shape

        # Second, crop out that area for running the entire model on
        f = rasterio.open(Path(REPO_DIR, self.data_fn), "r")
        src_profile = f.profile
        src_crs = f.crs.to_string()
        src_bounds = f.bounds
        transformed_mask_geom = fiona.transform.transform_geom("epsg:4326", src_crs, mask_geom)
        src_image, src_transform = rasterio.mask.mask(f, [transformed_mask_geom], crop=True, all_touched=True, pad=False)
        f.close()

        return src_image, src_profile, src_transform, shapely.geometry.shape(transformed_mask_geom).bounds, src_crs


class DataLoaderGlacier(DataLoader):
    @property
    def shapes(self):
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        self._shapes = value

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    def __init__(self, padding, path):
        self._padding = padding
        self._path = path

    def get_data_from_extent(self, extent):
        # transform the query extent to the source tiff's CRS
        source_img = rasterio.open(self._path)
        img_crs = source_img.meta["crs"]
        extent = extent_to_transformed_geom(extent, "EPSG:3857")
        extent = shapely.geometry.shape(extent)

        # extract that subwindow from the overall tiff
        bounds = extent.bounds
        window = from_bounds(
            left=extent.bounds[0],
            bottom=extent.bounds[1],
            right=extent.bounds[2],
            top=extent.bounds[3],
            transform=source_img.transform
        )

        return {
            "src_img": source_img.read(window=window),
            "src_crs": img_crs,
            "src_bounds": bounds,
            "src_transform": source_img.transform,
        }

    def get_data_from_shape_by_extent(self, extent, shape_layer):
        pass

    def get_data_from_shape(self, shape):
        pass

    def get_area_from_shape_by_extent(self, extent, shape_layer):
        pass

    def get_shape_by_extent(extent, shape_layer):
        pass
