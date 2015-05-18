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
    del merge['Unnamed: 0_x'], merge['# in sample'], \
        merge['Zip'], merge['Unnamed: 0_y'], merge['Zip/ZCTA'], merge['2010 Population']
    merge['Unemp. Rate'] = merge['Unemp. Rate'].str.strip('%')

    #merge.convert_objects(convert_numeric=True)

    #compute average unemployment
    #av_unemployment = merge['Unemp. Rate'].mean()
    #print av_unemployment

    #replace NaN values with zeroes
    #merge.fillna(0)


    #use average unemployment instead of Nan for ['Unemp. Rate'].
    #replace NaN values with zeroes
    #merge = merge.fillna(0)



    merge.to_csv(output_csv)
    return merge

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


def merge_demographic(dem_data, fin_data):
    """
    Read final dataset and append
    demographic data
    """
    df_final = pd.read_csv(fin_data)
    df_demo = pd.read_csv(dem_data)
    df_demo = df_demo[['GEO.id2', 'HD02_S03', 'HD02_S04', 'HD02_S05', 'HD02_S06', 'HD02_S07', 'HD02_S10', 'HD02_S23']]
    #remove labels in 1st row
    df_demo = df_demo.ix[1:]
    #extract zip
    #df_demo['GEO.display-label'] = df_demo['GEO.display-label'].str.strip('ZCTA5 ')
    #rename columns
    df_demo.columns = ['ZIP', 'White', 'Black', 'Indian', 'Native', 'Alaska', 'Asian', 'Other']
    #ZIP to int
    df_demo['ZIP'] = df_demo['ZIP'].astype(float)


    merge = pd.merge(df_final, df_demo, how='left', left_on='ZIP', right_on='ZIP')

    merge.to_csv("C:\BigData\Zemanta_challenge_1_data/part_two_new2.csv")
    return merge


def merge_age(age_data, fin_data):
    """
    Read final dataset and
    append age data
    """
    df_final = pd.read_csv(fin_data)
    df_demo = pd.read_csv(age_data)
    df_demo = df_demo[['GEO.id2', 'HD03_S01', 'SUBHD0201_S02', 'SUBHD0201_S03', 'SUBHD0201_S04', 'SUBHD0201_S05', 'SUBHD0201_S06', 'SUBHD0201_S07', 'SUBHD0201_S08',
                       'SUBHD0201_S09', 'SUBHD0201_S10', 'SUBHD0201_S11', 'SUBHD0201_S12', 'SUBHD0201_S13', 'SUBHD0201_S14', 'SUBHD0201_S15', 'SUBHD0201_S16', 'SUBHD0201_S17',
                       'SUBHD0201_S18', 'SUBHD0201_S19', 'SUBHD0201_S20']]
    #remove labels in 1st row
    df_demo = df_demo.ix[1:]
    #extract zip
    #df_demo['GEO.display-label'] = df_demo['GEO.display-label'].str.strip('ZCTA5 ')
    #rename columns
    df_demo.columns = ['ZIP', 'MalesPerFemales', 'Under5', '_5to9', '_10to14', '_14to19', '_20to24', '_25to29', '_30to34',
                       '_35to39', '_40to44', '_45to49', '_50to54', '_55to59', '_60to64', '_65to69', '_70to74', '_75to79', '_80to84', '_85to89', '_90over']

    df_demo['ZIP'] = df_demo['ZIP'].astype(float)
    merge = pd.merge(df_final, df_demo, how='left', left_on='ZIP', right_on='ZIP')
    merge = merge[pd.notnull(merge['Unemp. Rate'])]
    merge = merge.fillna(0)
    merge.to_csv("C:\BigData\Zemanta_challenge_1_data/FINAL_nan_new2.csv")
    return merge



unemployment_file = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
population_file = "C:\BigData\Zemanta_challenge_1_data/output.csv"
output_file = "C:\BigData\Zemanta_challenge_1_data/part_one.csv"
establishments_file = "C:\BigData\Zemanta_challenge_1_data/establishments_by_zip.dat"
establishments_out = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
training_set = "C:\BigData\Zemanta_challenge_1_data/training_set.tsv"
#partial = file_handler(unemployment_file, population_file, output_file)
#merge_establishments(establishments_file, establishments_out)
#zip_remover(training_set)
fin_data = "C:\BigData\Zemanta_challenge_1_data/part_one.csv"
dem_data = "C:\BigData\Zemanta_challenge_1_data/demographic.csv"
age_data = "C:\BigData\Zemanta_challenge_1_data/age.csv"
part_two = "C:\BigData\Zemanta_challenge_1_data/part_two_new2.csv"
#merge_demographic(dem_data, fin_data)
merge_age(age_data, part_two)

