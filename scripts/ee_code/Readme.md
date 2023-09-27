#### Files:

    get_training_images.py
    	Used to download the images used for training.
    	The image ids are specified in gdrive.yaml, train:image_ids
    	The satellite images after querying are saved in gdrive.yaml, train:gdrive_folder
    	Usage: python3 get_training_images.py -c gdrive.yaml

    get_inference_images.py
    	Used to download new images for a certain region given WRS path/row and date.
    	The wrs path/row for Landsat-7 images in the HKH region are specified in gdrive.yaml, infer:wrs_path_row
    	Downloads the images with least cloudcover within the region between dates infer:start_date and infer:end_date
    	The satellite images after querying are saved in gdrive.yaml, infer:gdrive_folder
    	Usage: python3 get_inference_images.py -c gdrive.yaml
    	
    utils.py
    	Contains function wrapper to perform operations on GEE