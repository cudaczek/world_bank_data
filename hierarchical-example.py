import os

import pandas
from sklearn import impute

from data.WorldBankDataLoader import WorldBankDataLoader
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import numpy as np

from existence_checker import get_region, verify_region_countries

word_bank_data_loader = WorldBankDataLoader()
all_countries = word_bank_data_loader.all_countries()
all_regions = word_bank_data_loader.regions

# this example clusters by population density & urban population in time
# missing values are replaced by mean values of a particular country's data
NULL_PLACEHOLDER = -1
imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='mean')
demografic_indicators = ['Population density (people per sq. km of land area)',
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


def get_data_per_country(indicators, strategy, region_name):
    data_per_country = {}
    region_data = get_region(region_name)
    region_data = verify_region_countries(indicators, region_data)
    for country_name in region_data:
        data = region_data[country_name][
            [*indicators]].fillna(NULL_PLACEHOLDER)
        imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy=strategy)
        data_per_country[country_name] = imputer.fit_transform(data)
    return data_per_country


if __name__ == "__main__":
    strategy = "mean"
    for region in all_regions:
        used_indicators = demografic_indicators
        data_per_country = get_data_per_country(used_indicators, strategy, region)

        featured_countries = [country for (country, data) in data_per_country.items() if data.shape == (60, 11)]

        X = np.array([np.array(data_per_country[country]).flatten() for country in featured_countries])
        metric = "euclidean"
        method = "ward"
        Z = linkage(X, method, metric=metric)
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
        plt.savefig("demography/dendrogram_regions/" + region + "_hierarchical_" + metric + "_" + method + "_" + strategy + ".png")
        plt.show()
