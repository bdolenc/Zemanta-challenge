__author__ = 'blaz'

import csv
from collections import OrderedDict

def file_handler(file1, file2, output_file):
    """open both files, iterate trough one
    and search by zip in the other. Paste
    result to new file"""

    #read files with csv reader
    unemployment_data = csv.reader(open(file1))
    population_density = csv.reader(open(file2))

    dict_unemployment = OrderedDict([(row[0], row[1]) for row in unemployment_data])
    dict_population = OrderedDict([(line[0], [line[1], line[2], line[3]]) for line in population_density])

    keys = dict_unemployment.keys()
    with open(output_file, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        for key in keys:
            if (key in dict_population.keys()):
                print dict_unemployment.get(key)
                #writer.writerows(key, dict_unemployment.get(key), dict_population.get(key))

    """
    with open(output_file, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows([key, dict_unemployment.get(key), ",".join(dict_population.get(key))] for key in keys)
    """





unemployment_file = "C:\BigData\Zemanta_challenge_1_data/unemployment_by_zip.csv"
population_file = "C:\BigData\Zemanta_challenge_1_data/population_densisty_area_by_zip.csv"
output_file = "C:\BigData\Zemanta_challenge_1_data/output.csv"
file_handler(unemployment_file, population_file, output_file)