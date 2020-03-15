# This code outputs different shp files for clean and debris labels 
# Usage: python3 test_get_clean_debris.py -f [shp_filename with debris information] -o [output_directory]
# python3 test_get_clean_debris.py -f ../data/vector_data/2005/nepal/data/Glacier_2005.shp -o ./
# Output:   The real debris labels are saved in [output_directory]/clean.shp
#           The snow_index debris labels are saved in [output_directory]/debris.shp
import argparse
import geopandas
import pathlib

def main(inp_filename, output_path):
    labels = geopandas.read_file(inp_filename)
    clean = labels[labels["Glaciers"] == "Clean Ice"]
    debris = labels[labels["Glaciers"] == "Debris covered"]
    clean.to_file(str(output_path)+'/clean.shp')
    debris.to_file(str(output_path)+'/debris.shp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--input_shp_file",
            type=str,
            help="Define input shp file to seperate debris and clean glaciers",
    )

    parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            help="Define output path to save files",
    )

    parsed_opts = parser.parse_args()
    output_path = pathlib.Path(parsed_opts.output_dir).resolve()
    if not output_path.exists():
        output_path.mkdir()
    inp_filename = parsed_opts.input_shp_file

    try:
        assert(inp_filename)
    except Exception as e:
        print("Input filename must be specified. Use flag -f")
        exit(0)
    
    main(inp_filename,output_path)
    print("Completed succesfully")