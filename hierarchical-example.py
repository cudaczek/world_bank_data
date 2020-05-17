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

DEMOGRAPHIC_INDICATORS = [
    'Population density (people per sq. km of land area)',
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
    "Population density (people per sq. km of land area)"
]

ECONOMIC_INDICATORS = [
    'GNI per capita',
    'Adjusted savings: education expenditure (% of GNI)',
    'GDP (current US$)',
    # 'Central government debt, total (% of GDP)',
    'Exports of goods and services (% of GDP)',
    'Imports of goods and services (% of GDP)',
    'Final consumption expenditure (% of GDP)',
    'Gross capital formation (% of GDP)',
    'Expense (% of GDP)',
    'Inflation, GDP deflator (annual %)'
]

SOCIODEMOGAPHIC_INDICATORS = list(WorldBankDataLoader().sociodemographic_indicators().values())


def get_data_per_country(indicators, region_name, group_name, start_year=None):
    region_data = get_region(region_name, group_name, start_year=start_year)
    region_data = verify_region_countries(indicators, region_data)
    return region_data


def plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators, strategy="mean", color_threshold=500):
    # after verification, skip all features that are missed by at least one country among those who passed the 30% verification
    features_to_skip = {}
    for country_name in data_per_country.keys():
        df1 = data_per_country[country_name]
        missing_features = df1.columns[df1.isna().all()].tolist()
        for missing_feature in missing_features:
            if not missing_feature in features_to_skip.keys():
                features_to_skip[missing_feature] = []
            features_to_skip[missing_feature].append(country_name)

    print("Skipping features:", features_to_skip)

    for country, country_data in data_per_country.items():
        country_data.drop(list(features_to_skip.keys()), axis=1, inplace=True)

    indicators = [ind for ind in used_indicators if ind not in features_to_skip]

    for country_name in data_per_country:
        data = data_per_country[country_name][[*indicators]].fillna(NULL_PLACEHOLDER)
        imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy=strategy)
        data_per_country[country_name] = imputer.fit_transform(data)

    X = np.array([np.array(data_per_country[country]).flatten('F') for country in featured_countries])
    metric = "euclidean"
    method = "ward"
    Z = linkage(X, method, metric=metric, optimal_ordering=True)
    fig = plt.figure(figsize=(80, 20))
    # (80, 20) for whole world
    # (25, 20)
    plt.title('Hierarchical Clustering Dendrogram ' + region, fontsize=20)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.yscale("log")
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        # leaf_font_size=10,  # font size for the x axis labels
        labels=np.array(featured_countries),
        color_threshold=color_threshold,
        ax=ax
    )
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.tick_params(axis='y', which='major', labelsize=10)
    plt.savefig(
        group_name + "/dendrogram_regions/" + region + "_hierarchical_" + metric + "_" + method + "_" + strategy + ".png")
    plt.show()


def evaluate_demography():
    group_name = "demography"
    for region in all_regions:
        if region != 'NA':
            used_indicators = DEMOGRAPHIC_INDICATORS
            if region != "NAC":
                data_per_country = get_data_per_country(used_indicators, region, group_name)
                featured_countries = [country for (country, data) in data_per_country.items()]
                plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators)
            if region == 'NAC':
                data_per_country_NAC = get_data_per_country(used_indicators, 'NAC', group_name)
                data_per_country = get_data_per_country(used_indicators, 'ECS', group_name)
                data_per_country.update(data_per_country_NAC)
                region = 'NAC&ECS'
                featured_countries = [country for (country, data) in data_per_country.items()]
                plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators)


def evaluate_economy():
    group_name = "economy"
    for region in all_regions:
        if region != 'NA':
            used_indicators = ECONOMIC_INDICATORS
            if region == 'NAC':
                data_per_country_NAC = get_data_per_country(used_indicators, 'NAC', group_name, start_year=1989)
                data_per_country = get_data_per_country(used_indicators, 'ECS', group_name, start_year=1989)
                data_per_country.update(data_per_country_NAC)
                featured_countries = [country for (country, data) in data_per_country.items()]
                region = 'ECS&NAC'
                plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators, color_threshold=100000000000)
            elif region != "ECS":
                data_per_country = get_data_per_country(used_indicators, region, group_name, start_year=1989)
                featured_countries = [country for (country, data) in data_per_country.items()]
                plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators, color_threshold=100000000000)


def evaluate_sociodemography():
    group_name = "sociodemography"
    for region in all_regions:
        if region != 'NA':
            used_indicators = SOCIODEMOGAPHIC_INDICATORS
            if region == 'NAC':
                data_per_country_NAC = get_data_per_country(used_indicators, 'NAC', group_name, start_year=1979)
                data_per_country = get_data_per_country(used_indicators, 'ECS', group_name, start_year=1979)
                data_per_country.update(data_per_country_NAC)
                featured_countries = [country for (country, data) in data_per_country.items()]
                region = 'ECS&NAC'
                plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators, color_threshold=160)
            elif region != "ECS":
                data_per_country = get_data_per_country(used_indicators, region, group_name, start_year=1979)
                featured_countries = [country for (country, data) in data_per_country.items()]
                plot_dendrogram(group_name, region, data_per_country, featured_countries, used_indicators, color_threshold=160)


def evaluate_demography_world():
    group_name = "demography"
    used_indicators = DEMOGRAPHIC_INDICATORS
    data_per_country = {}
    for region in all_regions:
        if region != 'NA':
            data_per_country.update(get_data_per_country(used_indicators, region, group_name))
    featured_countries = [country for (country, data) in data_per_country.items()]
    plot_dendrogram(group_name, "World_optimal", data_per_country, featured_countries, used_indicators)


def evaluate_sociodemography_world():
    group_name = "sociodemography"
    used_indicators = SOCIODEMOGAPHIC_INDICATORS
    data_per_country = {}
    for region in all_regions:
        if region != 'NA':
            data_per_country.update(get_data_per_country(used_indicators, region, group_name, start_year=1979))
    featured_countries = [country for (country, data) in data_per_country.items()]
    plot_dendrogram(group_name, "World_optimal", data_per_country, featured_countries, used_indicators)


def evaluate_economy_world():
    group_name = "economy"
    used_indicators = ECONOMIC_INDICATORS
    data_per_country = {}
    for region in all_regions:
        if region != 'NA':
            data_per_country.update(get_data_per_country(used_indicators, region, group_name, start_year=1989))
    featured_countries = [country for (country, data) in data_per_country.items()]
    plot_dendrogram(group_name, "World_optimal", data_per_country, featured_countries, used_indicators, color_threshold=100000000000)


if __name__ == "__main__":
    evaluate_demography_world()
    evaluate_sociodemography_world()
    evaluate_economy_world()
    evaluate_demography()
    evaluate_economy()
    evaluate_sociodemography()
