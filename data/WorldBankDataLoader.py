import wbdata


class WorldBankDataLoader:

    def __init__(self):
        self.init_regions()
        self.countries = ["POL", "ITA", "HU"]

    def all_countries(self):
        return wbdata.get_country(display=False)

    def init_regions(self):
        self.regions = {}
        for country in self.all_countries():
            region_id = country['region']['id']
            if region_id not in self.regions:
                self.regions.update({region_id: country['region']['value']})

    def demographic_indicators(self):
        return {
            'SP.DYN.CBRT.IN': 'Birth rate, crude (per 1,000 people)',
            'SP.DYN.CDRT.IN': 'Death rate, crude (per 1,000 people)',
            'SP.POP.TOTL.MA.ZS': 'Population, male (% of total population)',
            'SP.POP.BRTH.MF': 'Sex ratio at birth (male births per female births)',
            'SP.POP.DPND': 'Age dependency ratio (% of working-age population)',
            'SP.POP.DPND.OL': 'Age dependency ratio, old (% of working-age population)',
            'SP.POP.DPND.YG': 'Age dependency ratio, young (% of working-age population)',
            'SP.URB.TOTL.IN.ZS': 'Urban population (% of total population)',
            'SH.DYN.MORT': 'Mortality rate, under-5 (per 1,000 live births)',
            'SP.DYN.TFRT.IN': 'Fertility rate, total (births per woman)',
            'EN.POP.DNST': 'Population density (people per sq. km of land area)'
        }

    def demography(self, country=None):
        indicators = self.demographic_indicators()
        if country is None:
            country = self.countries
        return wbdata.get_dataframe(indicators, country=country, convert_date=False)
