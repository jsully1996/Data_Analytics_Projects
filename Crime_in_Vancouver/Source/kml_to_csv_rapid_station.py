from bs4 import BeautifulSoup
import csv
import os
import re

def process_coordinate_string(list_test):
    """
    Take the coordinate string from the KML file, and break it up into [Lat,Lon,Lat,Lon...] for a CSV row and other columns
    """
    ret = []

    ret.append(list_test[3])              # Name

    coordstr = list_test[2].rstrip('\n')
    space_splits = coordstr.split(" ")
    #take out the empty values
    space_splits = list(filter(None, space_splits))
#    # There was a space in between <coordinates>" "-80.123...... hence the [1:]
    for split in space_splits[:1]:
        comma_split = split.split(',')

        ret.append(comma_split[1])    # lat
        ret.append(comma_split[0])    # lng
        #to test the output: print(ret)

    return ret

def kml_to_csv(rootDir, kmlFile, csvFile):
    """
    Open the KML. Read the KML. Open a CSV file. Process a coordinate string to be a CSV row.
    """
    with open(rootDir + kmlFile, encoding='utf8') as f:
        dataz=f.read().replace('\n', '')

        # we need to clean via regex
        dataz = re.sub(r"<gx:labelVisibility>\d+</gx:labelVisibility>", "", dataz)
        s = BeautifulSoup(dataz, 'xml')
        with open(rootDir + csvFile, 'w', newline='', encoding='utf8') as csvfile:
            #Define the headers
            header = ['STATION', 'LAT', 'LONG']
            writer = csv.writer(csvfile)

            writer.writerow(header)
            total_list = []
            for placemark in s.find_all('Placemark'):
                #added conditions for no values in child tags
                name = placemark.find('name').string  \
                       if placemark.find('name') is not None  \
                       else 'None'

                description = placemark.find('description').string  \
                              if placemark.find('description') is not None \
                              else 'None'

                coords = placemark.find('coordinates').string \
                         if placemark.find('coordinates') is not None  \
                         else 'None'
                #create a list for and append values for each row
                list_test = []
                list_test.extend((name.string, description.string, coords.string))
                if placemark.find('ExtendedData'):
                   print('in before')
                   for item in placemark.find_all('SimpleData'):
                          print(item.string)
                          list_test.append((item.string))
                #total_list.append(list_test)
                print('listed', list_test)
                total_list.append(process_coordinate_string(list_test))
                #print(coords.string)
                #print(total_list)
            writer.writerows(total_list)

def main():
    #Define the absolute path
    abs_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)

    kml_to_csv(abs_path, \
               '/Big_Data_Project/data/skytrain_stations/rapid_transit_stations.kml',\
               '/Big_Data_Project/data/skytrain_stations/rapid_transit_stations.csv')

if __name__ == "__main__":
    main()
