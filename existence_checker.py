import datetime
import matplotlib.pyplot as plt
import os
import pandas
from data.WorldBankDataLoader import WorldBankDataLoader

word_bank_data_loader = WorldBankDataLoader()
all_countries = word_bank_data_loader.all_countries()
all_regions = word_bank_data_loader.regions

DEMOGRAPHIC_INDICATORS = ['Population density (people per sq. km of land area)',
                          'Urban population (% of total population)',
                          "Birth rate, crude (per 1,000 people)",
                          "Death rate, crude (per 1,000 people)",
                          "Population, male (% of total population)",
                          "Sex ratio at birth (male births per female births)",
                          "Age dependency ratio (% of working-age population)",
                          "Age dependency ratio, old (% of working-age population)",
                          "Age dependency ratio, young (% of working-age population)",
                          "Urban population (% of total population)",
                          "Mortality rate, under-5 (per 1,000 live births)",
                          "Fertility rate, total (births per woman)",
                          "Population density (people per sq. km of land area)"]


def get_region(region_name):
    region_countries = {country['name']: None for country in all_countries if
                        country['region']['id'] == region_name}
    print(region_name)
    to_del = []
    for country_name in region_countries:
        try:
            country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
            region_countries[country_name] = pandas.read_csv(country_data_path)
        except FileNotFoundError:  # thrown for regions because could not call get_dataframe for region
            print("Downloading data for %s failed." % country_name)
            to_del.append(country_name)
        except TypeError:
            print("Downloading data for %s failed." % country_name)
            to_del.append(country_name)
    for country_name in to_del:
        region_countries.pop(country_name)
    return region_countries


def verify_country(country_data, indicators, name, maximum_allowed_missing_values=0.3):
    not_exist_counter = 0
    all_counter = 0
    for indicator in indicators:
        for index, row in country_data[["date", indicator]].iterrows():
            if pandas.isnull(row[indicator]):
                not_exist_counter += 1
            all_counter += 1
    percentage = not_exist_counter / all_counter
    if percentage <= maximum_allowed_missing_values:
        return True, percentage
    else:
        return False, percentage


def verify_region_countries(indicators, countries):
    ok_countries = dict()
    for country_name in list(countries.keys()):
        country_data = countries[country_name]
        is_successful, percentage = verify_country(country_data, indicators, country_name)
        if is_successful:
            ok_countries.update({country_name: country_data})
    return ok_countries


def verify_region_countries_and_plot_statistics(indicators, countries, region_name):
    results = dict()
    for country_name in list(countries.keys()):
        country_data = countries[country_name]
        is_successful, percentage = verify_country(country_data, indicators, country_name, 0)
        print(country_name)
        results.update({country_name: percentage})
    X = results.keys()
    y = results.values()
    plt.subplots(figsize=(20, 20))
    plt.title(region_name)
    plt.bar(X, y)
    plt.xticks(range(len(X)), X)
    plt.xticks(rotation=45)
    plt.axhline(y=0.3, color='r', linestyle='-')
    plt.savefig(os.path.join("demography", "country_verification",
                             datetime.datetime.now().strftime("%m-%d-%Y_%H_%M_%S") + '_verified_region_statistics.png'))
    plt.show()


if __name__ == "__main__":
    for region_name in all_regions:
        if region_name != 'NA':
            print(region_name)
            region_countries = get_region(region_name)

            verify_region_countries_and_plot_statistics(indicators=DEMOGRAPHIC_INDICATORS, countries=region_countries,
                                                        region_name=region_name)
