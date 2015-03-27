import pandas as pd


def file_handler(file1, file2, output_csv):
    """
    Read both csv files and merge them by ZIP
    """
    df_unemployment = pd.read_csv(file1)
    df_population = pd.read_csv(file2)

    merge = pd.merge(df_population, df_unemployment, how = 'left', left_on = 'Zip/ZCTA', right_on = 'Zip')

    merge.to_csv(output_csv)


unemployment_file = "C:\BigData\Zemanta_challenge_1_data/unemployment_by_zip.csv"
population_file = "C:\BigData\Zemanta_challenge_1_data/population_densisty_area_by_zip.csv"
output_file = "C:\BigData\Zemanta_challenge_1_data/output.csv"
file_handler(unemployment_file, population_file, output_file)