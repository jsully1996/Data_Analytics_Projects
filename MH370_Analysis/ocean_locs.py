import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
import geopandas as gp
from mpl_toolkits.basemap import Basemap # use for plotting world maps
from geopy.distance import vincenty
from geopy.exc import GeopyError
import geopy.geocoders
from geopy.distance import great_circle
import utm

file = open('Data/Sattelite/*.txt') 
lines = []
for i in file.readlines():
    if '100   ' in i:
        i.rstrip('\n')
        lines.append(i)                
def utm_to_latlong(east, north, zone_number = 10, zone_letter = 'U'):
    try:
        return utm.to_latlon(east, north, zone_number, zone_letter)
    except:
        print ("You might need to install utm.. 'pip install utm'")       

def latlong_to_utm(lat, long):
    try:
        return utm.from_latlon(lat, long)
    except:
        print ("You might need to install utm.. 'pip install utm'")

sat_height = 42170 #m
elevation_angle = np.radians(40) #elevation angle of satellite
earth_radius = 6371 #in km

"""
Computes the ping arc distance from the satellite to the plane.
Returns the angle in degrees.
"""
def satellite_calc(radius,orbit,angle):
    interim = (np.sqrt(-radius**2+orbit**2*(1./np.cos(angle)**2))-orbit*np.tan(angle))/np.float(orbit+radius)
    return np.degrees(2*np.arctan(interim))
ping_arc = satellite_calc(earth_radius,sat_height,elevation_angle)
dist_sat = earth_radius*np.radians(satellite_calc(earth_radius,sat_height,elevation_angle))
circle_pts = great_circle(ping_arc,360,64.5,0)
circle_lats = []
circle_lons = []

for i in range(len(circle_pts)):
    circle_lats.append(circle_pts[i][0])
for i in range(len(circle_pts)):
    circle_lons.append(circle_pts[i][1])

err1 = 0.8*ping_arc
err2 = 1.2*ping_arc

circle_pts_err1 = great_circle(err1,360,64.5,0)
circle_pts_err2 = great_circle(err2,360,64.5,0)
circle_lon_err1 = []
for i in range(len(circle_pts_err1)):
    circle_lon_err1.append(circle_pts_err1[i][1])
    
circle_lon_err2 = []
for i in range(len(circle_pts_err2)):
    circle_lon_err2.append(circle_pts_err2[i][1])
    
circle_lat_err1 = []
for i in range(len(circle_pts_err1)):
    circle_lat_err1.append(circle_pts_err1[i][0])
    
circle_lat_err2 = []
for i in range(len(circle_pts_err2)):
    circle_lat_err2.append(circle_pts_err2[i][0])

fig = plt.figure(figsize=[21,15])
# setup  Conformal basemap.
fig = Basemap(width=10000000,height=8000000,projection='lcc',resolution='c',lat_0=5,lon_0=90.,suppress_ticks=True)
fig.drawcoastlines()
fig.drawmapboundary(fill_color='lightblue')
fig.fillcontinents(color='#FFD39B',lake_color='lightblue')
parallels = np.arange(-50.,50,10.)
fig.drawparallels(np.arange(-50,50,10),labels=[1,1,0,1], fontsize=15)
meridians = np.arange(50.,130.,10.)
fig.drawmeridians(np.arange(50,130,10),labels=[1,1,0,1], fontsize=15)
x4,y4 = fig(99.8,6.35) #show Lankawi for electrical fire scenario
x5,y5 = fig(circle_lons,circle_lats)
x6,y6 = fig(circle_lon_err1,circle_lat_err1)
x7,y7 = fig(circle_lon_err2,circle_lat_err2)

fig.plot(x4,y4,'go',markersize=10,label='Lankawi Island')

#draw circle showing extent of Inmarsat sat radar detection
fig.plot(x5,y5,'r-',markersize=5,label='Inferred MH370 Location from Satellite, 5th Ping')
fig.plot(x6,y6,'r--',markersize=5,label='with 5,10,20% error')
fig.plot(x7,y7,'r--',markersize=5)

#draw arrows showing flight path
arrow1 = plt.arrow(x2[0],y2[0],x2[1]-x2[0],y2[1]-y2[0],linewidth=3,color='blue',linestyle='dashed',label='flight path')
arrow2 = plt.arrow(x2[1],y2[1],x2[2]-x2[1],y2[2]-y2[1],linewidth=3,color='blue',linestyle='dashed',label='flight path')

legend = plt.legend(loc='lower left',fontsize=10,frameon=True,title='Legend',markerscale=1,prop={'size':15})
legend.get_title().set_fontsize('20')
plt.title('Inmarsat Ping Estimation', fontsize=30)
plt.show()