import pandas as pd
import numpy as np
import csv

def file_handler(file1, file2, output_csv):
    """
    Read both csv files and merge them by ZIP
    """
    df_unemployment = pd.read_csv(file1)
    df_population = pd.read_csv(file2)

    merge = pd.merge(df_unemployment, df_population, how='left', left_on='ZIP', right_on='Zip/ZCTA')
    del merge['Unnamed: 0_x'], merge['Land-Sq-Mi'], merge['# in sample'], merge['Zip'], merge['Unnamed: 0_y'], merge['Zip/ZCTA'], merge['2010 Population']
    #merge['Unemp. Rate'] = merge['Unemp. Rate'].map(lambda x: x.strip('%'))
    merge['Unemp. Rate'] = merge['Unemp. Rate'].str.strip('%')
    merge.to_csv(output_csv)


def merge_establishments(input_name, output_name):
    """
    Read chunks of establishments_by_zip in pandas
    data frame and append to appropriate zip in
    data frame
    """
    # new column names
    column_names = ['ZIP', '_all', '_1to4', '_5to9', '_10to19', '_20to49', '_50to99', '_100to249',
                    '_250to499', '_500to999', '_1000plus']
    #dataframe for optimized data
    df_total = pd.DataFrame(data=np.zeros((0, len(column_names))), columns=column_names)
    df_other = pd.DataFrame(data=np.zeros((0, 1)), columns=['ZIP'])
    #iterate through establishments_by_zip
    with open(input_name, "rb") as csv_file:
        data = csv.reader(csv_file, delimiter='|')
        next(data, None)  # skip header
        #process each line and append to df_merged
        curr_zip = 10000
        index = -1
        curr_sector = 'empty'
        list_df = []
        first_iter = True
        for row in data:
            #Total for all sectors
            #print row
            if row[6] == 'Total for all sectors':
                #check if zip already in df
                if row[1] != curr_zip:
                    index += 1
                    #add zip to df_total
                    df_total = df_total.append({'ZIP': row[1]}, ignore_index='true')
                    #add all establishments number
                    df_total.set_value(index, '_all', row[11])
                    curr_zip = row[1]
                    col_i = 2
                #add other data to a curr zip
                else:
                    df_total.set_value(index, column_names[col_i], row[11])
                    col_i += 1
            else:
                if row[6] != curr_sector:
                    curr_sector = row[6]
                    list_df.append(df_other)
                    df_other = pd.DataFrame(data=np.zeros((0, 2)), columns=['ZIP', curr_sector])
                    print curr_sector
                    index = 0
                    df_other = df_other.append({'ZIP': row[1]}, ignore_index='true')
                    df_other.set_value(index, curr_sector, row[11])
                    curr_zip = row[1]
                    #print df_other
                else:
                    if row[10] == 'All establishments':
                        if row[1] != curr_zip:
                            index += 1

                            df_other = df_other.append({'ZIP': row[1]}, ignore_index='true')
                            df_other.set_value(index, curr_sector, row[11])
                            curr_zip = row[1]
                        else:
                            df_other.set_value(index, curr_sector, row[11])

    list_df.append(df_other)
    for df in list_df:
        df_total = pd.merge(df_total, df, how='outer', on='ZIP')
    #print df_total
    df_total.to_csv(output_name)


def zip_remover(training_set):
    """
    Remove zipcodes not
    in learning data.
    """
    df_training = pd.read_csv(training_set, sep='\t')
    print len(set(df_training['zip']))


unemployment_file = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
population_file = "C:\BigData\Zemanta_challenge_1_data/output.csv"
output_file = "C:\BigData\Zemanta_challenge_1_data/final_data.csv"
establishments_file = "C:\BigData\Zemanta_challenge_1_data/establishments_by_zip.dat"
establishments_out = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
training_set = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
#file_handler(unemployment_file, population_file, output_file)
#merge_establishments(establishments_file, establishments_out)
zip_remover(training_set)