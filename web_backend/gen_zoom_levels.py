import gdal2tiles

def main():
    gdal2tiles.generate_tiles('/home/aortiz/research_projects/glacier_mapping/web_tool/tiles/hkh_byte.tif', '/home/aortiz/research_projects/glacier_mapping/web_tool/tiles/hkh/', zoom='15-17')

if __name__ == '__main__':
    main()