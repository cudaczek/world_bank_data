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

    def sociodemographic_indicators(self):
        return {
            'SL.UEM.TOTL.ZS': 'Unemployment, total (% of total labor force) (modeled ILO estimate)',
            # 'SH.STA.MALN.ZS': 'Prevalence of underweight, weight for age (% of children under 5)',
            # 'SE.ADT.LITR.ZS': 'Literacy rate, adult total (% of people ages 15 and above)',
            'SP.POP.1564.TO.ZS': 'Population ages 15-64 (% of total population)',
            'SP.POP.0014.TO.ZS': 'Population ages 0-14 (% of total population)',
            'SP.POP.65UP.TO.ZS': 'Population ages 65 and above (% of total population)',
            'SH.IMM.MEAS': 'Immunization, measles (% of children ages 12-23 months)',
            'SH.IMM.IDPT': 'Immunization, DPT (% of children ages 12-23 months)'
        }

    def economic_indicators(self):
        return {
            'NY.GNP.PCAP.CD': 'GNI per capita',
            'NY.ADJ.AEDU.GN.ZS': 'Adjusted savings: education expenditure (% of GNI)',
            'NY.GDP.MKTP.CD': 'GDP (current US$)',
            # 'GC.DOD.TOTL.GD.ZS': 'Central government debt, total (% of GDP)',
            'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',
            'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
            'NE.CON.TOTL.ZS': 'Final consumption expenditure (% of GDP)',
            'NE.GDI.TOTL.ZS': 'Gross capital formation (% of GDP)',
            'GC.XPN.TOTL.GD.ZS': 'Expense (% of GDP)',
            'NY.GDP.DEFL.KD.ZG': 'Inflation, GDP deflator (annual %)'
        }

    def demography(self, country=None):
        indicators = self.demographic_indicators()
        if country is None:
            country = self.countries
        return wbdata.get_dataframe(indicators, country=country, convert_date=False)

    def sociodemography(self, country=None):
        indicators = self.sociodemographic_indicators()
        if country is None:
            country = self.countries
        return wbdata.get_dataframe(indicators, country=country, convert_date=False)

    def economy(self, country=None):
        indicators = self.economic_indicators()
        if country is None:
            country = self.countries
        return wbdata.get_dataframe(indicators, country=country, convert_date=False)
