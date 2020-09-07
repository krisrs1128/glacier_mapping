"""
Output prediction tiffs on all tiffs in a directory

python3 -m predict_tiffs -d $DATA_DIR/pipeline_demo/demo_data/data/2005/nepal/ -m $DATA_DIR/runs/minimal_run/models/model_final.pt
"""
from addict import Dict
import argparse
import pathlib
import yaml
import glacier_mapping.infer as gmi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict over tiles in a directory")
    parser.add_argument("-d", "--tile_dir", type=str)
    parser.add_argument("-m", "--model_path", type=str, default="./")
    parser.add_argument("-c", "--train_yaml", type=str, default="../../conf/train.yaml")
    parser.add_argument("-p", "--postprocess_yaml", type=str, default="../../conf/postprocess.yaml")
    parser.add_argument("-o", "--output_dir", type=str, default="output")
    args = parser.parse_args()

    # load the model and setup the output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model = gmi.load_model(args.train_yaml, args.model_path)

    # loop over input tiles and make predictions
    input_tiles = list(pathlib.Path(args.tile_dir).glob("*.tif*"))
    for path in input_tiles:
        img, x, y_hat = gmi.predict_tiff(path, model, conf_path=args.postprocess_yaml)
        gmi.write_geotiff(y_hat, img.meta, output_dir / path.stem + ".tiff")
