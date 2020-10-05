## Configurations files for channel selection experiment and Debris Vs. Glaciers experiment.

The files numbers/names correspond to the experiment number/name

## Experiments for selecting channels
All expriemnts are carried on images with more than 10% of unified glacier class

Exp Number | channels | Notes
  --- | --- | ---
0 | 10, 11, 12, 13 | selected via random forest from channels of higher than 10% importance
1 | 4, 1, 3 | B5 B2 B3 (Recommeded from ICMOD)
2 | 4, 1, 3, 13, 14 | B5 B2 B3 + Elevation + Slope
3 | 4, 1, 3, 10, 11, 12 | B5 B2 B3 + NDWI + NDSI + NDVI
4 | 4, 1, 3, 10, 11, 12, 13, 14 | B5 B2 B3 + NDWI + NDSI + NDVI + Elevation + Slope
5 | 4, 1, 3, 13 | B5 B2 B3 + Elevation
6 | 4, 1, 3, 14 | B5 B2 B3 + Slope
7 | 0, 1, 2 | RGB
8 | 0, 1, 2, 13, 14 | RGB + Elevation + Slope
9 | 0, 1, 2, 10, 11, 12 | RGB + NDWI + NDSI + NDVI
10| 0, 1, 2, 10, 11, 12, 13, 14 | RGB + NDWI + NDSI + NDVI + Elevation + Slope
11| 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | L7 channels
12| 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14 | L7 channels + Elevation + Slope
13| 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | L7 channels + NDWI + NDSI + NDVI
14| 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14| L7 channels + NDWI + NDSI + NDVI + Elevation + Slope (Best configuration)
15| 14,13,11,7,12,5,10 | selected via random forest from channels of higher than 5% importance

## Experiments for choosing multi-class vs binary model
All expriemnts are carried on images with more than 0% of both classes
Exp Name | Out Channels
  --- | ---
 all | channel 0: unified glaciers
 clean | channel 1: clean glaciers
 debris | channel 2: debris glaciers
 clean_debris | channels 1, 2: multi-class glaciers