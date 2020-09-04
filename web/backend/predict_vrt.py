import glacier_mapping.inference as gmi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict over tiles in a directory")
    parser.add_argument("-d", "--tile_dir", type=str)
    parser.add_argument("-o", "--model_path", type=str, default="./")
    parser.add_argument("-o", "--model_yaml", type=str, default="conf/train.yaml")
    parser.add_argument("-n", "--output_dir", type=str, default="output.vrt")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # loads an empty model, without weights
    train_conf = Dict(yaml.safe_load(open(args.model_yaml, "r")))
    model = Framework(torch.nn.BCEWithLogitsLoss(), train_conf.model_opts, train_conf.optim_opts).model

    # if GPU is available, inference will be faster
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    input_tiles = list(pathlib.Path(args.tile_dir).glob("*.tiff"))
    for path in input_tiles:
        img, x, y_hat = gmi.predict_tiff(path, model)
        write_geotiff(y_hat, img.meta, output_dir / path.basename)

