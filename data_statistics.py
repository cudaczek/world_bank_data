import matplotlib.pyplot as plt
import os
import pandas
from data.WorldBankDataLoader import WorldBankDataLoader
from textwrap import wrap

if __name__ == "__main__":

    # get countries in region - here it's Europe & Central Asia
    all_countries = WorldBankDataLoader().all_countries()
    europe_and_central_asia_countries = {country['name']: None for country in all_countries if
                                         country['region']['id'] == 'ECS'}
    for country_name in europe_and_central_asia_countries:
        country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
        europe_and_central_asia_countries[country_name] = pandas.read_csv(country_data_path)

    INDICATORS = ['Population density (people per sq. km of land area)',
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

    data_per_country = set()

    for country_name in list(europe_and_central_asia_countries.keys()):
        data = europe_and_central_asia_countries[country_name]
        xs = {}
        for indicator in INDICATORS:
            lista = []
            last_year_found = 2019
            last_year = 2020
            for index, row in data[["date", indicator]].iterrows():
                if pandas.isnull(row[indicator]):
                    last_year = int(row["date"])
                else:
                    if last_year_found - int(row["date"]) != 0:
                        lista.append((int(row["date"]), last_year - int(row["date"])))
                        last_year_found = int(row["date"])
            lista.reverse()
            xs.update({indicator: lista})

        fig, ax = plt.subplots(figsize=(30, 20))
        facecolors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:olive', 'tab:gray', 'tab:pink', 'tab:brown','tab:cyan','tab:orange','tab:green']
        for i, facecolor in enumerate(facecolors):
            ax.broken_barh(xs[INDICATORS[i]], ((i+1)*10, 9), facecolors=facecolor)
        ax.set_xlabel('years')

        ax.set_xlim(1959, 2020)
        ax.set_yticks([15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135])
        ax.set_xticks(range(1960, 2020, 5))
        ax.set_yticklabels(['\n'.join(wrap(ind, 20)) for ind in INDICATORS])
        ax.grid(True)
        plt.title(country_name)
        plt.savefig(os.path.join("demography", "null_data", country_name + ".png"))
        # plt.show()
        plt.clf()
        plt.close()
        print("saved "+country_name)