
import utm

def utm_to_latlong(easting, northing, zone_number = 10, zone_letter = 'U'):
    return utm.to_latlon(easting, northing, zone_number, zone_letter)


def main(input, output):
    crime_fp = open(input, 'r')
    clean_fp = open(output, 'w')

    for line in crime_fp:
        cols = line.strip().split(',')
        crime_type, year, month, day, hour, minute, hundred_block, nhood, utm_x, utm_y = cols

        if utm_x != '0' and utm_y != '0':   #to remove type "offence against a person"

            if crime_type != 'TYPE':
                x = float(utm_x)
                y = float(utm_y)
                lat, lon = utm_to_latlong(x, y)
                latitude = str(lat)
                longitude = str(lon)
            else:
                latitude = 'LATITUDE'
                longitude = 'LONGITUDE'

            clean_fp.write(crime_type +','+ year +','+ month +', '+ day +', '+ hour +', '+ minute +', '+ hundred_block +','+ nhood +','+ latitude +','+ longitude +'\n')

    crime_fp.close()
    clean_fp.close()


if __name__ == '__main__':
    #input = sys.argv[1]
    input = '/Users/jaideepmishra/downloads/Big_Data_Project/data/crime_csv_all_years.csv'
    #output = sys.argv[2]
    output = '/Users/jaideepmishra/downloads/Big_Data_Project/data/crime_test.csv'
    main(input, output)
