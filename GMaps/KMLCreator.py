#The code bellow is based on Sample python code published on
#https://developers.google.com/kml/articles/csvtokml?csw=1
#but highly modified.
#The code is published under MIT license.

import xml.dom.minidom
import pandas as pd


def createPlacemark(kmlDoc, row, aver_row, all_row):
    # This creates a  element for a row of data.
    # A row is a dict.
    placemarkElement = kmlDoc.createElement('Placemark')
    extElement = kmlDoc.createElement('ExtendedData')
    placemarkElement.appendChild(extElement)

    latitude = row[1]['latitude']
    longitude = row[1]['longitude']
    colors = ['#FFFF00', '#FFCC00', '#FF3300', '#CCFF00', '#CC9900', '#CC3300', '#99FF00', '#999900', '#993300'
              , '#000099', '#660099', '#66CC99', '#FF0099', '#FFFFFF', '#000000', '#003300', '#CCCCCC', '#CC00CC'
              , '#669966', '#663366']

    #create popup with relevant info about zip
    dataElement = kmlDoc.createElement('Data')
    dataElement.setAttribute('name', 'ZIP')
    valueElement = kmlDoc.createElement('value')
    dataElement.appendChild(valueElement)
    valueText = kmlDoc.createTextNode(str(row[1]['ZIP']))
    valueElement.appendChild(valueText)
    extElement.appendChild(dataElement)

    #add Average title
    dataElement = kmlDoc.createElement('Data')
    dataElement.setAttribute('name', 'CLUSTER STATS')
    valueElement = kmlDoc.createElement('value')
    dataElement.appendChild(valueElement)
    valueText = kmlDoc.createTextNode('/')
    valueElement.appendChild(valueText)
    extElement.appendChild(dataElement)

    #add cluster average to popup
    for column in aver_row.columns.values:
        dataElement = kmlDoc.createElement('Data')
        dataElement.setAttribute('name', column)
        valueElement = kmlDoc.createElement('value')
        dataElement.appendChild(valueElement)
        valueText = kmlDoc.createTextNode(str(aver_row.iloc[0][column]))
        valueElement.appendChild(valueText)
        extElement.appendChild(dataElement)

    #add ZIP STATS title
    dataElement = kmlDoc.createElement('Data')
    dataElement.setAttribute('name', 'ZIP STATS')
    valueElement = kmlDoc.createElement('value')
    dataElement.appendChild(valueElement)
    valueText = kmlDoc.createTextNode('/')
    valueElement.appendChild(valueText)
    extElement.appendChild(dataElement)

    #add zip stats to popup
    column_names = column_names = ['White', 'Black', 'Unemp. Rate', 'Density Per Sq Mile']
    for column in column_names:
        dataElement = kmlDoc.createElement('Data')
        dataElement.setAttribute('name', column)
        valueElement = kmlDoc.createElement('value')
        dataElement.appendChild(valueElement)
        valueText = kmlDoc.createTextNode(str(all_row.iloc[0][column]))
        valueElement.appendChild(valueText)
        extElement.appendChild(dataElement)

    #create placemark, color based on label
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


def createKML(df_data, fileName, aver_file, all_file):
    # This constructs the KML document from the CSV file.
    kmlDoc = xml.dom.minidom.Document()

    kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
    kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
    kmlElement = kmlDoc.appendChild(kmlElement)
    documentElement = kmlDoc.createElement('Document')
    documentElement = kmlElement.appendChild(documentElement)

    df_averages = pd.read_csv(aver_file)
    df_averages = df_averages.drop([df_averages.columns[0]], axis=1)

    df_all = pd.read_csv(all_file)

    for row in df_data.iterrows():
        # if row[1]['Labels'] == 19:
        aver_row = df_averages.loc[df_averages['Labels'] == row[1]['Labels']]
        all_row = df_all.loc[df_all['ZIP'] == row[1]['ZIP']]
        placemarkElement = createPlacemark(kmlDoc, row, aver_row, all_row)
        documentElement.appendChild(placemarkElement)
    kmlFile = open(fileName, 'w')
    kmlFile.write(kmlDoc.toprettyxml('  ', newl='\n', encoding='utf-8'))


def prepare_data(coordinates, hc_results):
    df_coor = pd.read_csv(coordinates)
    header_row = ['ZIP', 'Labels']
    df_hc = pd.read_csv(hc_results, names=header_row)
    #print df_coor['zip']
    df_coor['zip'] = df_coor['zip'].astype(float)
    df_merged = pd.merge(df_hc, df_coor, how='left', left_on='ZIP', right_on='zip')
    return df_merged


all_file = "C:\BigData\Zemanta_challenge_1_data/FINAL_nan_new2.csv"
df_data = prepare_data('zipcode.csv', '../hc_results.csv')
createKML(df_data, 'google-points.kml', '../averages_per_cluster_final.csv', all_file)



