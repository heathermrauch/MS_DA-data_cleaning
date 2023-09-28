import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

FILE_PATH = r'C:\Users\user\OneDrive\Documents\Education\Western Govenors University\MS - Data Analytics\Data ' \
            r'Cleaning\medical_raw_data.csv'
OUTPUT_PATH = r'..\output'
DTYPES = {
    'CaseOrder': int, 'Customer_id': str, 'Interaction': str, 'UID': str, 'City': str, 'State': str, 'County': str,
    'Zip': str, 'Lat': float, 'Lng': float, 'Population': int, 'Area': str, 'Timezone': str, 'Job': str,
    'Children': float, 'Age': float, 'Education': str, 'Employment': str, 'Income': float, 'Marital': str,
    'Gender': str, 'ReAdmis': str, 'VitD_levels': float, 'Doc_visits': int, 'Full_meals_eaten': int, 'VitD_supp': int,
    'Soft_drink': str, 'Initial_admin': str, 'HighBlood': str, 'Stroke': str, 'Complication_risk': str,
    'Overweight': float, 'Arthritis': str, 'Diabetes': str, 'Hyperlipidemia': str, 'BackPain': str, 'Anxiety': float,
    'Allergic_rhinitis': str, 'Asthma': str, 'Services': str, 'Initial_days': float, 'TotalCharge': float,
    'Additional_charges': float, 'Item1': int, 'Item2': int, 'Item3': int, 'Item4': int, 'Item5': int, 'Item6': int,
    'Item7': int, 'Item8': int
}


def configure_logging(output_path: str):
    """Configures logging for the program. Output is directed to both the console and a log file.

    :param output_path: The path to which the log file should be stored.
    :return: None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_path),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame

    :param file_path: a str representing the path at which the CSV file to be read can be located
    :return: a pandas DataFrame object containing the contents of the CSV located at the given file path
    """
    return pd.read_csv(file_path, index_col='Unnamed: 0', dtype=DTYPES, na_values=pd.NA)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Inspects a pandas DataFrame for duplicate rows and removes them if necessary

    :param df: a pandas DataFrame which should be inspected for duplicate rows
    :return: df with all duplicate rows removed, if any existed
    """
    n_dups = df[df.duplicated()].shape[0]
    if n_dups > 0:
        logging.info(f'{n_dups} duplicates found.\nDropping duplicate records.')
        df.drop_duplicates(inplace=True)
        deduplicate(df)
    else:
        logging.info('No duplicates found.')
    return df


def impute_missing(df: pd.DataFrame, methods: dict) -> pd.DataFrame:
    """Imputes missing values in a pandas DataFrame according to the provided map

    :param df: a pandas DataFrame with values to be imputed
    :param methods: a dict with the structure {column name: imputation method}. Acceptable imputation methods include
    'mean', 'median', and 'mode'.
    :return: pandas DataFrame with imputed values
    """
    for col, method in methods.items():
        logging.info(f'Imputing {col} with the {method}')
        if method == 'mean':
            df[f'{col}_imputed'] = df[col].isna()
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            df[f'{col}_imputed'] = df[col].isna()
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mode':
            df[f'{col}_imputed'] = df[col].isna()
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            raise NotImplementedError(f'{method} is not an implemented imputation method.')
    return df


def detect_outliers(df: pd.DataFrame, query: dict) -> pd.DataFrame:
    """Inspects each numeric column of a pandas DataFrame for outliers

    :param df: a pandas DataFrame which should be inspected for outliers
    :param query: a dict of search strings to be used to find outliers
    :return: df with z-score and outlier columns added for each numeric column
    """
    # Add z-score and outlier columns
    for col, query_string in query.items():
        df[f'{col}_zscore'] = stats.zscore(df[col])
        df[f'{col}_outlier'] = df.index.isin(df.query(query_string).index)
        outliers = df[f'{col}_outlier'].sum()
        if outliers > 0:
            logging.info(f'{col} contains {outliers} outliers')
    return df


def float_to_yn(x: float) -> str:
    """Used to transform yes/no fields read in as floats

    :param x: 1. or 0.
    :return: 'Yes' or 'No'
    """
    if x == 1.:
        return 'Yes'
    elif x == 0.:
        return 'No'


def main():
    date_time = f'{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
    log_dir = os.path.join(OUTPUT_PATH, date_time)
    log_path = os.path.join(log_dir, f'{date_time}.log')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(log_dir)
    configure_logging(log_path)

    filename = __file__.split('\\')[-1]
    logging.info(f'{5 * "*"} Running {filename} {5 * "*"}')

    logging.info(f'Loading medical_data from {FILE_PATH}')
    medical_data = load_data(FILE_PATH)

    logging.info(f'{5 * "*"} Detecting duplicate rows {5 * "*"}')
    deduplicate(medical_data)

    logging.info('Plotting quantitative variables with missing values...')
    fields = [['Children', 'Age'], ['Income', 'Initial_days']]
    plt.close()
    fig, ax = plt.subplots(2, 2)
    fig.set_tight_layout(True)
    fig.suptitle('Numeric variables with missing values')
    for r, row in enumerate(fields):
        for c, col in enumerate(row):
            ax[r, c].hist(medical_data[col])
            ax[r, c].set_title(col)
    plt.savefig(os.path.join(log_dir, 'initial_dist.png'))

    logging.info(f'{5 * "*"} Imputing missing values {5 * "*"}')
    methods = {
        'Children': 'median',
        'Age': 'mean',
        'Income': 'median',
        'Soft_drink': 'mode',
        'Overweight': 'mode',
        'Anxiety': 'mode',
        'Initial_days': 'median'
    }
    medical_data_imputed = impute_missing(medical_data.copy(), methods)

    logging.info('Verifying distributions pre and post imputation...')
    fields = ['Children', 'Age', 'Income', 'Initial_days']
    plt.close()
    fig, ax = plt.subplots(4, 2, figsize=(6.4, 9))
    fig.set_tight_layout(True)
    fig.suptitle('Pre/Post imputation distibutions')
    for f, field in enumerate(fields):
        ax[f, 0].hist(medical_data[field])
        ax[f, 0].set_title(f'{field} pre')
        ax[f, 1].hist(medical_data_imputed[field])
        ax[f, 1].set_title(f'{field} post')
    plt.savefig(os.path.join(log_dir, 'dist_compare.png'))

    # Align dtypes with data dictionary
    # performing prior to imputation threw errors
    int_cols = ['Children', 'Age']
    for col in int_cols:
        medical_data_imputed[col] = medical_data_imputed[col].astype(int)
    yn_cols = ['Overweight', 'Anxiety']
    for col in yn_cols:
        medical_data_imputed[col] = medical_data_imputed[col].apply(float_to_yn, convert_dtype=True)

    logging.info(f'{5 * "*"} Detecting outliers {5 * "*"}')
    outlier_query = {
        'Lat': 'Lat_zscore > 3 | Lat_zscore < -3',
        'Lng': 'Lng_zscore < -3',
        'Population': 'Population_zscore > 3',
        'Children': 'Children_zscore > 3',
        'Income': 'Income_zscore > 3',
        'VitD_levels': 'VitD_levels_zscore > 3 | VitD_levels_zscore < -3',
        'Full_meals_eaten': 'Full_meals_eaten_zscore > 3',
        'VitD_supp': 'VitD_supp_zscore > 3',
        'TotalCharge': 'TotalCharge_zscore > 3',
        'Additional_charges': 'Additional_charges_zscore > 3'
    }
    medical_data_outliers = detect_outliers(medical_data_imputed.copy(), outlier_query)

    logging.info(f'{5 * "*"} Handling outliers {5 * "*"}')
    logging.info('Excluding records where State = "PR"')
    logging.info('Excluding Population outliers')
    logging.info('Keeping all other outliers')
    medical_data_excluded = medical_data_outliers.query('State == "PR" | Population_outlier')
    medical_data_clean = medical_data_outliers.query('State != "PR" & not Population_outlier')

    logging.info('Reviewing distributions before and after handling outliers...')
    fields = ['Lat', 'Lng', 'Population']
    plt.close()
    fig, ax = plt.subplots(3, 2, figsize=(8, 9))
    fig.suptitle('medical_data distributions')
    fig.set_tight_layout(True)
    for r, field in enumerate(fields):
        ax[r, 0].hist(medical_data_outliers[field])
        ax[r, 0].set_title(field + ' before handling outliers')
        ax[r, 1].hist(medical_data_clean[field])
        ax[r, 1].set_title(field + ' after handling outliers')
    plt.savefig(os.path.join(log_dir, 'post_outlier_dist.png'))

    logging.info(f'{5 * "*"} Run Complete {5 * "*"}')


if __name__ == '__main__':
    main()
