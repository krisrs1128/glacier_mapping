The python files included in the shared folder are as follows:
  - test_plot_band_intensity.py
  - ml_classifier.py
  - ml_predict.py
  - ml_plot.py
  - test_calculate_iou.py
  - test_get_clean_debris.py

# Machine Learning Pipeline
The files are run in the following order:
  1. test_plot_band_intensity.py (creates csv input file)
  2. ml_classifier.py (used for gridsearch or train)
  3. ml_predict.py (predict the output images for trained model)
  4. ml_plot.py (get .png file)

## test_plot_band_intensity.py
This code is used to plot the band intensity for each of the classes for channels 1,2,3,4,5 and 7 in Landsat 7 image. This code also outputs data_[pattern for shp_filename].csv in the current folder. The pixels in the csv file and intensity plot are preprocessed according to the rules described in function check_condition.

### Usage: 
```python3 test_get_clean_debris.py -f [shp_filename with debris information] -p [pattern for shp_filenames] -o [output filename]```
### Example 
```python3 test_plot_band_intensity.py -f ../data/vector_data/2005/nepal/data/Glacier_2005.shp -p LE07_140041_20051012* -o test.png```
### Output:   
The output is saved as test.png in the current directory

## ml_classifier.py
This code can be used to train or grid search different machine learning models on the given csv data file. The labels are assumed to be clean, debris, background. The best parameters for training needs to be updated after running grid search. The default values for [csv_data_file] and [output_folder_location] are data.csv and ./saved_models respectively.

### Usage: 
Train: ```python3 ml_classifier.py -f [csv_data_file] -o [output_folder_location] -t```
Grid Search: ```python3 ml_classifier.py -f [csv_data_file] -o [output_folder_location] -gs```
### Example 
```python3 ml_classifier.py -f data.csv -o ./saved_models -t```
### Output:   
After training, the models are saved in the [output_folder_location] as svm_linear.pkl, svm_rbf.pkl, mlp.pkl, and decision_tree.pkl respectively.

## ml_predict.py
This code can be used to predict labels for given image using different saved machine learning models. The image numpy files are taken from the slices folder. The profram looks for saved models in the ./saved_models folder and saves the output numpy files in the ./inference_data/ folder. Change the image_filename to predict on new slices.

### Usage: 
```python3 ml_predict.py```
### Output:   
After prediction, the numpy files are saved in the ./inference_data/ folder as decision_tree_output.npy, mlp_output.npy, svm_linear_output.npy, svm_rbf_output.npy.

# Individual files

## test_calculate_iou.py
This code calculates the IOU between debris labels and pseudo debris labels for each slice of the specified image filename.
### Usage: 
```python3 test_calculate_iou.py -f [shp_filename with debris information] -p [pattern for corresponding landsat image filename]```
### Example 
```python3 test_calculate_iou.py -f ../data/vector_data/2005/nepal/data/Glacier_2005.shp -p LE07_140041_*```
### Output:   
The real debris labels are saved in ```./temp_files/real/filename```
The snow_index debris labels are saved in ```./temp_files/snow/filename ```

## test_get_clean_debris_iou.py
This code outputs different shp files for clean and debris labels 
### Usage: 
```python3 test_get_clean_debris.py -f [shp_filename with debris information] -o [output_directory]```
### Example 
```python3 test_get_clean_debris.py -f ../data/vector_data/2005/nepal/data/Glacier_2005.shp -o ./```
### Output:   
The real debris labels are saved in ```[output_directory]/clean.shp```
The snow_index debris labels are saved in ```[output_directory]/debris.shp```
