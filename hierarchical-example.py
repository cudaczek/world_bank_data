import os

import pandas
from sklearn import impute

from data.WorldBankDataLoader import WorldBankDataLoader
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import numpy as np

all_countries = WorldBankDataLoader().all_countries()
europe_and_central_asia_countries = {country['name']:None for country in all_countries if country['region']['id'] == 'ECS'}
for country_name in europe_and_central_asia_countries:
    country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
    europe_and_central_asia_countries[country_name] = pandas.read_csv(country_data_path)

# this example clusters by population density & urban population in time
# missing values are replaced by mean values of a particular country's data
NULL_PLACEHOLDER = -1
imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='mean')


if __name__ == "__main__":
    data_per_country = {}
    for country_name in europe_and_central_asia_countries:
        data = europe_and_central_asia_countries[country_name][
                ['Population density (people per sq. km of land area)','Urban population (% of total population)']].fillna(NULL_PLACEHOLDER)
        data_per_country[country_name] = imputer.fit_transform(data)

    featured_countries = [country for (country, data) in data_per_country.items() if data.shape==(60,2)]

    X = np.array([np.array(data_per_country[country]).flatten() for country in featured_countries])

    Z = linkage(X, 'ward')
    plt.figure(figsize=(25, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
    )
    plt.savefig("hierarchical.png")
    plt.show()
