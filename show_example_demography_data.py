import os
import pprint

import pandas
from sklearn import cluster, impute, preprocessing
import itertools
from data.WorldBankDataLoader import WorldBankDataLoader
from plot_clusters import plot_with_pca, plot_with_tsne
from existence_checker import get_region, verify_region_countries


def get_data_for_region_per_country(region_name):
    all_countries = WorldBankDataLoader().all_countries()
    selected_countries = {country['name']: None for country in all_countries if country['region']['id'] == region_name}
    for country_name in selected_countries:
        country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
        selected_countries[country_name] = pandas.read_csv(country_data_path)
    return selected_countries


def replace_nulls(country_data, used_indicators):
    NULL_PLACEHOLDER = -1
    imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='median')
    data_per_country = {}
    for country_name in country_data:
        data = country_data[country_name][[*used_indicators]].fillna(NULL_PLACEHOLDER)
        data_per_country[country_name] = imputer.fit_transform(data)
    return data_per_country



def cluster_and_plot(group_name, countries_data_to_show, used_indicators):
    countries_data_to_show = verify_region_countries(used_indicators, countries_data_to_show)

    # after verification, skip all features that are missed by at least one country among those who passed the 30% verification
    features_to_skip = {}
    for country_name in countries_data_to_show.keys():
        df1 = countries_data_to_show[country_name]
        missing_features = df1.columns[df1.isna().all()].tolist()
        for missing_feature in missing_features:
            if not missing_feature in features_to_skip.keys():
                features_to_skip[missing_feature]=[]
            features_to_skip[missing_feature].append(country_name)

    print("Skipping features:", features_to_skip)

    for country, country_data in countries_data_to_show.items():
        country_data.drop(list(features_to_skip.keys()), axis=1, inplace=True)

    used_indicators = [ind for ind in used_indicators if ind not in features_to_skip]

    # end of skipping missing features

    dates = next(iter(countries_data_to_show.values()))['date'].to_numpy()  # any country would suffice, we just need dates

    data_per_country_without_nulls = replace_nulls(countries_data_to_show, used_indicators)

    # cluster countries with k-means
    clusterer = cluster.KMeans(n_clusters=15, random_state=42)
    # filter out countries without all expected features
    featured_countries = [country for (country, data) in data_per_country_without_nulls.items()]
    # scale feature; flatten arrays for K-means
    X = [preprocessing.scale(data_per_country_without_nulls[country], axis=0).flatten(order='F') for country in
        featured_countries]

    labels = clusterer.fit_predict(X)
    grouped_countries = {label: [] for label in labels}
    for i in range(len(featured_countries)):
        grouped_countries[labels[i]].append(featured_countries[i])

    # save results of clustering
    with open(group_name+".txt", "w", encoding="utf-8") as fout:
        fout.write(pprint.pformat(grouped_countries))

    feature_names = [indicator + "(" + str(year) + ")" for (indicator, year) in itertools.product(used_indicators, dates)]

    # display results of clustering by:
    #  bringing data into 2 dimensions via PCA
    #plot_with_pca(X, labels, featured_countries, len(used_indicators), feature_names)
    # using t-SNE
    plot_with_tsne(X, labels, featured_countries, perplexity=5, iterations=1000, learning_rate=15, should_save=True, filename=group_name)

regions = ['ECS', 'NAC', 'LCN', 'MEA', 'SSF', 'SAS', 'EAS']

group_name = 'sociodemography'
countries_data_to_show =  get_region(regions[0], group_name, start_year=1979) 
for region in regions[1:]:
    countries_data_to_show.update(get_region(region, group_name, start_year=1979) )
used_indicators = list(WorldBankDataLoader().sociodemographic_indicators().values())
cluster_and_plot(group_name, countries_data_to_show, used_indicators)

group_name = 'demography'
countries_data_to_show =  get_region(regions[0], group_name) 
for region in regions[1:]:
    countries_data_to_show.update(get_region(region, group_name))
used_indicators = list(WorldBankDataLoader().demographic_indicators().values())
cluster_and_plot(group_name, countries_data_to_show, used_indicators)

group_name = 'economy'
countries_data_to_show =  get_region(regions[0], group_name, start_year=1989) 
for region in regions[1:]:
    countries_data_to_show.update(get_region(region, group_name, start_year=1989) )
used_indicators = list(WorldBankDataLoader().economic_indicators().values())
cluster_and_plot(group_name, countries_data_to_show, used_indicators)