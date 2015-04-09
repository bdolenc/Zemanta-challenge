import pandas as pd
import numpy as np
import csv

def file_handler(file1, file2, output_csv):
    """
    Read both csv files and merge them by ZIP
    """
    df_unemployment = pd.read_csv(file1)
    df_population = pd.read_csv(file2)

    merge = pd.merge(df_population, df_unemployment, how = 'left', left_on = 'Zip/ZCTA', right_on = 'Zip')

    merge.to_csv(output_csv)


def merge_establishments(input_name, output_name):
    """
    Read chunks of establishments_by_zip in pandas
    dataframe and append to appropriate zip in dataframe
    """
    #new column names
    column_names = ['ZIP', '_all', '_1to4', '_5to9', '_10to19', '_20to49', '_50to99', '_100to249',
                    '_250to499', '_500to999', '_1000plus']

    #dataframe for optimized data
    df_merged = pd.DataFrame(data=np.zeros((0, len(column_names))), columns=column_names)

    #iterate through establishments_by_zip
    with open(input_name, "rb") as csv_file:
        data = csv.reader(csv_file, delimiter='|')
        next(data, None) #skip header
        #process each line and append to df_merged
        first_zip = 10000
        index = -1
        for row in data:
            #check if zip already in df
            print row
            if row[1] != first_zip:
                index += 1
                #add zip to df_merged
                df_merged = df_merged.append({'ZIP': row[1]}, ignore_index='true')
                #add all establishments number
                df_merged.set_value(index, '_all', row[11])
                first_zip = row[1]
                col_i = 2
            #add other data to a curr zip
            else:
                df_merged.set_value(index, column_names[col_i], row[11])
                col_i += 1


    #df_merged = df_merged.append({'ZIP': 50234}, ignore_index='true')
    #df_merged._5to9[df_merged.ZIP == 50234] = 5
    print df_merged


    #print df_establishments

unemployment_file = "C:\BigData\Zemanta_challenge_1_data/unemployment_by_zip.csv"
population_file = "C:\BigData\Zemanta_challenge_1_data/population_densisty_area_by_zip.csv"
output_file = "C:\BigData\Zemanta_challenge_1_data/output.csv"
establishments_file = "C:\BigData\Zemanta_challenge_1_data/establishments_test.csv"
establishments_out = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
#file_handler(unemployment_file, population_file, output_file)
merge_establishments(establishments_file, establishments_out)