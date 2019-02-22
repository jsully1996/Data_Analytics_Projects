import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
import geopandas as gp
from mpl_toolkits.basemap import Basemap # use for plotting world maps
file = open('runways.txt') 
lines = []
for i in file.readlines():
    if '100   ' in i:
        i.rstrip('\n')
        lines.append(i)                
print ("Total number of documented landing possibilities:", len(lines))
def get_coordinates(lines):
    coord1 = []
    coord2 = []
    coord3 = []
    coord4 = []
    for i in lines:
        parts = i.split()
        coord1.append(parts[9])
        coord2.append(parts[10])
        coord3.append(parts[18])
        coord4.append(parts[19])
    return coord1,coord2,coord3,coord4
coords = np.zeros((len(lines), 4))
z = get_coordinates(lines)
for i in range(len(z[0])):
    coords[i][0] = z[i]
    coords[i][1] = z[i]
    coords[i][2] = z[i]
    coords[i][3] = z[i]
lats = []
lons = []
for i in range(len(coords)):
    lats.append( (coords[i][0]+coords[i][2])/2.0 )
    lons.append( (coords[i][1]+coords[i][3])/2.0 )
coord_avg = np.zeros((len(coords), 2))

for i in range(len(lons)):
    coord_avg[i][0] = lats[i]
    coord_avg[i][1] = lons[i]

coords_len = np.zeros((len(coords),1))
earth_radius = 6371000 #in m, http://en.wikipedia.org/wiki/Earth
"""
Haversine equation.
"""
def haversine(r,lat1,lon1,lat2,lon2):
    dist = 2.0*r*np.arcsin(np.sqrt(np.sin(np.radians(lat2-lat1)/2.0)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon1)/2.0)**2))
    return dist  
for i in range(len(coords_len)):
    coords_len[i] = haversine(earth_radius,coords[i][0],coords[i][1],coords[i][2],coords[i][3])
    
min_run_dist = 5000* 0.3048 # A 777  needs at least a 5000 ft runway length to land.

runway_indices = []
for i in range(len(coords_len)):
    if coords_len[i] >= 1524.0 :
        runway_indices.append(i)
runway_coords = coords[runway_indices]
runway_avg = coords_avg[runway_indices]
# plot the plane's known positions
inmarsat_coord = [0, 64.5]
#Kuala Lumpur International Airport Coordinates: http://www.distancesfrom.com/my/Kuala-Lumpur-Airport-(KUL)-Malaysia-latitude-longitude-Kuala-Lumpur-Airport-(KUL)-Malaysia-latitude-/LatLongHistory/3308940.aspx
kualalumpur_coord = [2.7544829, 101.7011363]
#Pulau Perak coordinates: http://tools.wmflabs.org/geohack/geohack.php?pagename=Pulau_Perak&params=5_40_50_N_98_56_27_E_type:isle_region:MY
pulauperak_coord = [5.680556,98.940833]
# Igari Waypoint. Source: # http://www.fallingrain.com/waypoint/SN/IGARI.html Given in hours,minutes,seconds.
igariwaypoint_coord = [6. + 56./60. + 12./3600., 103. + 35./60. + 6./3600.] 
runway_lats = []
runway_lons = []
# split coordinates into list form now to follow plotting example
for i in range(len(runway_avg)):
    runway_lats.append(runway_avg[i][0])
    runway_lons.append(runway_avg[i][1])

plane_lats = [2.7544829,(6.+56./60.+12./3600.),5.680556]
plane_lons = [101.7011363,(103.+35./60.+6./3600.),98.940833]
satellite_lats = 0
satellite_lons = 64.5
fig = plt.figure(figsize=[21,15])

# setup  Conformal basemap.
fig = Basemap(width=10000000,height=8000000,projection='lcc',resolution='c',lat_0=10,lon_0=90.,suppress_ticks=True) 
fig.drawcoastlines()
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)
#Runway Locs
x,y = fig(runway_lons,runway_lats)
#Known 777 Locs
x2,y2 = fig(plane_lons,plane_lats)
#Inmarsat Satellite Loc
x3,y3 = fig(satellite_lons,satellite_lats)
# plot coords w/ filled circles
fig.plot(x,y,'ko',markersize=5,label='Landable Runways')
fig.plot(x2,y2,'bo',markersize=10,label='MH370 Flight Path')
fig.plot(x3,y3,'ro',markersize=10,label='Inmarsat 3-F1')

arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

legend = plt.legend(loc='lower center',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')
plt.title('Landable Runways for a Boeing 777', fontsize=30)
plt.show()