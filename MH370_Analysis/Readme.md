Just playing around with data to improve my data analysis skills by exploring a new front...trying to figure out possible locations of the Malaysian Airlines flight MH370 which went missing in 2014.

##Dependencies
numpy 
scipy 
matplotlib
geopandas 
basemap 
geopy
utm

##Data
http://marine.ga.gov.au/#/
Go to dowloads and select :
1) MH370: 150m Bathymetry
2) MH370 : Backscatter
Extract all tif files

##Analysis
**SCENARIO 1** :The MH-370 landed safely somewhere on land.
There are only a limited number of ways that a Boeing-777 aircraft can land and that includes having a runway with a particular minimum length

We have a csv file containing data on all runways of the planet....We will filter out only those runways which have the minimum length for accomodating a 777 aircraft
We further filter runways based on the flight path which can be determined through satellite logs.

**SCENARIO 2** :The MH-370 landed somewhere in the ocean.
This is a little complex and required use of Regression model to determine the possible change in flight path given the backtraces of the satellite logs calculated using a 20% error and the co-ordinates matrix extracted from the tif files.
Please install scipy before running mcmodel.py
 