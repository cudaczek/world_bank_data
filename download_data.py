import datetime
import os

from data.WorldBankDataLoader import WorldBankDataLoader


def download_demography_data():
    wb_data_loader = WorldBankDataLoader()
    print(datetime.datetime.now())  # 2020-03-08 15:43:21.965790
    current_filename = os.path.abspath(os.path.dirname(__file__))
    demography_dict = "demography/downloaded_countries/"
    for country in wb_data_loader.all_countries():  # 304 countries
        country_id = country["id"]
        country_name = country["name"]
        try:
            demography_data = wb_data_loader.demography(country_id)

            store_location = os.path.join(current_filename, demography_dict, country_name + ".csv")
            demography_data.to_csv(store_location)
        except TypeError:  # thrown for regions because could not call get_dataframe for region
            print("Downloading data for %s failed." % country_name)
    print(datetime.datetime.now())  # 2020-03-08 16:06:20.442193


def download_sociodemography_data():
    wb_data_loader = WorldBankDataLoader()
    print(datetime.datetime.now())
    current_filename = os.path.abspath(os.path.dirname(__file__))
    for country in wb_data_loader.all_countries():
        country_id = country["id"]
        country_name = country["name"]
        try:
            sociodemography_data = wb_data_loader.sociodemography(country_id)

            store_location = os.path.join(current_filename, 'sociodemography', 'downloaded_countries',
                                          country_name + ".csv")
            sociodemography_data.to_csv(store_location)
        except TypeError:  # thrown for regions because could not call get_dataframe for region
            print("Downloading data for %s failed." % country_name)
    print(datetime.datetime.now())  # 2020-03-08 16:06:20.442193


def download_economic_data():
    wb_data_loader = WorldBankDataLoader()
    print(datetime.datetime.now())  # 2020-03-08 15:43:21.965790
    current_filename = os.path.abspath(os.path.dirname(__file__))
    economic_dict = "economy/downloaded_countries/"
    for country in wb_data_loader.all_countries():  # 304 countries
        country_id = country["id"]
        country_name = country["name"]
        try:
            economic_data = wb_data_loader.economy(country_id)
            store_location = os.path.join(current_filename, economic_dict, country_name + ".csv")
            economic_data.to_csv(store_location)
        except TypeError:  # thrown for regions because could not call get_dataframe for region
            print("Downloading data for %s failed." % country_name)
    print(datetime.datetime.now())  # 2020-03-08 16:06:20.442193


if __name__ == "__main__":
    # download_demography_data()
    download_sociodemography_data()
    # download_economic_data()
