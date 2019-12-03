#### Folder Structure Google Drive

The files can be found in: https://drive.google.com/open?id=1UQ2QSKkt-hcZOwzCpwZ7rWnURptEx-0F

    EEImages/
    	Nepal/
		2000/
		2010/
		Slope/
		Elevation/
	Bhutan/
		2000/
		2010/
		Slope/
		Elevation/

- 2010 GEE images are corrected
- usedForLabeling folder has the exact images used for creating labels
- otherImages folder has the remaining images
- (If this structure is not present, all the images are the ones used for labelling)
- 2010 pre-corrected images are the images with errors in them
- 2010 post-corrected images are the ones to use
- Since slope and elevation data is not temporal, we download them only once using the codes to download 2000 images

#### Files present

	landsat-7-images-used-for-labelling-2000-nepal (Download actual images used for labelling in 2000 for Nepal with elevation and slopes)
	landsat-7-images-used-for-labelling-2010-nepal (Download actual images used for labelling in 2010 for Nepal)
	landsat-7-remaining-images-2000-bhutan (Download images in 2000 for Bhutan with elevation and slopes)
	landsat-7-remaining-images-2000-nepal (Download images in 2000 for Nepal with elevation and slopes)
