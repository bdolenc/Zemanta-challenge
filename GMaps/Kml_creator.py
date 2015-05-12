import csv
import xml.dom.minidom
import sys
import pandas as pd


def extractAddress(row):
    # This extracts an address from a row and returns it as a string. This requires knowing
    # ahead of time what the columns are that hold the address information.
    return '%s,%s,%s,%s,%s' % (row['Address1'], row['Address2'], row['City'], row['State'], row['Zip'])


def createPlacemark(kmlDoc, row):
    # This creates a  element for a row of data.
    # A row is a dict.
    placemarkElement = kmlDoc.createElement('Placemark')
    extElement = kmlDoc.createElement('ExtendedData')
    placemarkElement.appendChild(extElement)

  # Loop through the columns and create a  element for every field that has a value.
    #for key in row:
    #    if key != ''
    #        dataElement = kmlDoc.createElement('Data')
    #        dataElement.setAttribute('name', key)
    #        valueElement = kmlDoc.createElement('value')
    #        dataElement.appendChild(valueElement)
    #        valueText = kmlDoc.createTextNode(row[key])
    #        valueElement.appendChild(valueText)
    #        extElement.appendChild(dataElement)
    latitude = row[1]['latitude']
    longitude = row[1]['longitude']
    colors = ['#FFFF00', '#FFCC00', '#FF3300', '#CCFF00', '#CC9900', '#CC3300', '#99FF00', '#999900', '#993300'
            ,'#000099', '#660099', '#66CC99', '#FF0099', '#FFFFFF', '#000000', '#003300', '#CCCCCC', '#CC00CC'
            ,'#669966', '#663366']

    pointElement = kmlDoc.createElement('Point')
    placemarkElement.appendChild(pointElement)
    coordinates = str(longitude) + ',' + str(latitude)
    coorElement = kmlDoc.createElement('coordinates')
    coorElement.appendChild(kmlDoc.createTextNode(coordinates))
    pointElement.appendChild(coorElement)
    colorElement = kmlDoc.createElement('color')
    colorElement.appendChild(kmlDoc.createTextNode(colors[row[1]['Labels']]))
    pointElement.appendChild(colorElement)
    styleUrl = kmlDoc.createElement('styleUrl')
    styleUrl.appendChild(kmlDoc.createTextNode('#' + str(row[1]['Labels'])))
    placemarkElement.appendChild(styleUrl)
    return placemarkElement


def createKML(df_data, fileName):
    # This constructs the KML document from the CSV file.
    kmlDoc = xml.dom.minidom.Document()

    kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
    kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
    kmlElement = kmlDoc.appendChild(kmlElement)
    documentElement = kmlDoc.createElement('Document')
    documentElement = kmlElement.appendChild(documentElement)

    # Skip the header line.
    #csvReader.next()

    for row in df_data.iterrows():
        placemarkElement = createPlacemark(kmlDoc, row)
        documentElement.appendChild(placemarkElement)
    kmlFile = open(fileName, 'w')
    kmlFile.write(kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8'))


def prepare_data(coordinates, hc_results):
    df_coor = pd.read_csv(coordinates)
    header_row = ['ZIP', 'Labels']
    df_hc = pd.read_csv(hc_results, names=header_row)
    print df_coor['zip']
    df_coor['zip'] = df_coor['zip'].astype(float)
    df_merged = pd.merge(df_hc, df_coor, how='left', left_on='ZIP', right_on='zip')
    return df_merged



def main():
    # This reader opens up 'google-addresses.csv', which should be replaced with your own.
    # It creates a KML file called 'google.kml'.

    # If an argument was passed to the script, it splits the argument on a comma
    # and uses the resulting list to specify an order for when columns get added.
    # Otherwise, it defaults to the order used in the sample.

    df_data = prepare_data('zipcode.csv', '../hc_results.csv')
    kml = createKML(df_data, 'google-points.kml')

    """
    if len(sys.argv) > 1:
        order = sys.argv[1].split(',')
    else:
        order = ['Office','Address1','Address2','Address3','City','State','Zip','Phone','Fax']
    csvreader = csv.DictReader(open('google-addresses.csv'), order)
    kml = createKML(csvreader, 'google-addresses.kml', order)
    """
if __name__ == '__main__':
    main()