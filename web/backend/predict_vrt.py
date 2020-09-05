import glacier_mapping.inference as gmi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict over tiles in a directory")
    parser.add_argument("-d", "--tile_dir", type=str)
    parser.add_argument("-o", "--model_path", type=str, default="./")
    parser.add_argument("-o", "--train_yaml", type=str, default="conf/train.yaml")
    parser.add_argument("-n", "--output_dir", type=str, default="output.vrt")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model = gmi.load_model(args.train_yaml, args.model_path)

    input_tiles = list(pathlib.Path(args.tile_dir).glob("*.tiff"))
    for path in input_tiles:
        img, x, y_hat = gmi.predict_tiff(path, model)
        gmi.write_geotiff(y_hat, img.meta, output_dir / path.basename)

