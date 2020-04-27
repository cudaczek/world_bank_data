import datetime
import os

from data.WorldBankDataLoader import WorldBankDataLoader

if __name__ == "__main__":
    wb_data_loader = WorldBankDataLoader()
    print(datetime.datetime.now())  # 2020-03-08 15:43:21.965790
    current_filename = os.path.abspath(os.path.dirname(__file__))
    demography_dict = "demography/downloaded_countries/"
    for country in wb_data_loader.all_countries():  # 304 countries
        country_id = country["id"]
        country_name = country["name"]
        try:
            demography_data = wb_data_loader.demography(country_id)

            store_location = os.path.join(current_filename, country_name + ".csv")
            demography_data.to_csv(store_location)
        except TypeError:  # thrown for regions because could not call get_dataframe for region
            print("Downloading data for %s failed." % country_name)
        break
    print(datetime.datetime.now())  # 2020-03-08 16:06:20.442193
