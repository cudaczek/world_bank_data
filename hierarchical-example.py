import os

import pandas
from sklearn import impute

from data.WorldBankDataLoader import WorldBankDataLoader
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import numpy as np

all_countries = WorldBankDataLoader().all_countries()
europe_and_central_asia_countries = {country['name']: None for country in all_countries if
                                     country['region']['id'] == 'ECS'}
for country_name in europe_and_central_asia_countries:
    country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
    europe_and_central_asia_countries[country_name] = pandas.read_csv(country_data_path)

# this example clusters by population density & urban population in time
# missing values are replaced by mean values of a particular country's data
NULL_PLACEHOLDER = -1
imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='mean')
used_indicators = ['Population density (people per sq. km of land area)',
                   'Urban population (% of total population)',
                   "Birth rate, crude (per 1,000 people)",
                   "Death rate, crude (per 1,000 people)",
                   "Population, male (% of total population)",
                   "Sex ratio at birth (male births per female births)",
                   "Age dependency ratio (% of working-age population)",
                   "Age dependency ratio, old (% of working-age population)",
                   "Age dependency ratio, young (% of working-age population)",
                   "Mortality rate, under-5 (per 1,000 live births)",
                   "Fertility rate, total (births per woman)"]


def get_data_per_country(indicators, strategy):
    data_per_country = {}
    for country_name in europe_and_central_asia_countries:
        data = europe_and_central_asia_countries[country_name][
            [*indicators]].fillna(NULL_PLACEHOLDER)
        imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy=strategy)
        data_per_country[country_name] = imputer.fit_transform(data)
    return data_per_country


if __name__ == "__main__":
    data_per_country = get_data_per_country(used_indicators, 'median')

    featured_countries = [country for (country, data) in data_per_country.items() if data.shape == (60, 11)]

    X = np.array([np.array(data_per_country[country]).flatten() for country in featured_countries])

    Z = linkage(X, 'ward')
    plt.figure(figsize=(25, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.yscale("log")
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        labels=np.array(featured_countries),
        color_threshold=500
    )
    plt.savefig("hierarchical.png")
    plt.show()
