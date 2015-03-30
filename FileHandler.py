import pandas as pd
import numpy as np

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
    df_establishments = pd.read_csv(input_name, delimiter='|')
    df_merged = pd.DataFrame(data=np.zeros((0, 3)), columns=['ZIP', 'biz', '5 to 9'])
    df_merged = df_merged.append({'ZIP':50234}, ignore_index='true')
    df_merged.biz[df_merged.ZIP == 50234] = 5
    print df_merged


    #print df_establishments




unemployment_file = "C:\BigData\Zemanta_challenge_1_data/unemployment_by_zip.csv"
population_file = "C:\BigData\Zemanta_challenge_1_data/population_densisty_area_by_zip.csv"
output_file = "C:\BigData\Zemanta_challenge_1_data/output.csv"
establishments_file = "C:\BigData\Zemanta_challenge_1_data/establishments_test.csv"
establishments_out = "C:\BigData\Zemanta_challenge_1_data/output_test.csv"
#file_handler(unemployment_file, population_file, output_file)
merge_establishments(establishments_file, establishments_out)