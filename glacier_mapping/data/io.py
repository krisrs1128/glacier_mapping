import pathlib
import numpy as np
from skimage.io import imsave


def write_npys(arrays, names=None, out_dir="."):
    """

    :param arrays: A list of arrays.
    :param out_dir: The directory to which we should write the arrays.
    :channels: A list of lists giving which channels to write each time
    """
    out_dir = pathlib.Path(out_dir)
    if not names:
        names = range(len(arrays))

    for array in arrays:
        np.save(array, out_dir / f"{names[i]}.npy")


def convert_png(npy_paths, channels=[[0]]):
    for i, path in enumerate(npy_paths):
        for ix in channels:
            npy = np.load(npy_paths[i])
            ix_str = "-".join([str(s) for s in channels])
            out_path = npy_paths[i].replace(".npy", f"{ix_str}.png")
            imsave(npy[:, :, ix], out_path)


def write_geotiff(y_hat, meta, output_path):
    """
    Write predictions to geotiff

    :param y_hat: A numpy array of predictions.
    :type y_hat: np.ndarray
    """
    # create empty raster with write geographic information
    dst_file = rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=y_hat.shape[0],
        width=y_hat.shape[1],
        count=y_hat.shape[2],
        dtype=np.float32,
        crs=meta["crs"],
        transform=meta["transform"]
    )

    y_hat = 255.0 * y_hat.astype(np.float32)
    for k in range(y_hat.shape[2]):
        dst_file.write(y_hat[:, :, k], k + 1)
