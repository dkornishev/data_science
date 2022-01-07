import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

print(os.getcwd())
DATA_CSV_PATH = os.getcwd() + "/../../resources/suicide_data.csv"

PER_100k = "suicides/100k pop"
SUICIDES_NO = "suicides_no"
POPULATION = "population"


def load_csv_data() -> DataFrame:
    loaded_data = pd.read_csv(DATA_CSV_PATH)

    # Prepend 0 to fix age sorting
    loaded_data.loc[loaded_data.age == "5-14 years", "age"] = "0" + loaded_data["age"]
    return loaded_data


def age_based_analysis(data: DataFrame):
    by_age = data.groupby(data.age)[[SUICIDES_NO, POPULATION]].sum()
    by_age["rate"] = by_age[SUICIDES_NO] / by_age[POPULATION] *100000
    x_axis = [x.removesuffix(" years") for x in by_age.index.values]

    plt.bar(x=x_axis, height=by_age["rate"], width=0.7, color="red")
    plt.xlabel("Age")
    plt.ylabel("Suicide rate per 100k")
    plt.show()


def country_based_analysis(data: DataFrame) -> Dict[str, int]:
    by_country = data.loc[data["year"] > 2010].groupby([data.country])[
        [SUICIDES_NO]].sum().sort_values(by=[SUICIDES_NO])

    return by_country.to_dict()


def population_size_to_suicide_rate_correlation(data: DataFrame) -> int:
    return data[[POPULATION, PER_100k]].corr().iloc[0][1]


def gdp_to_suicide(data: DataFrame) -> int:
    return data[["gdp_per_capita ($)", PER_100k]].corr().iloc[0][1]


def trend_across_years(data: DataFrame):
    data_by_year = data.groupby(data["year"])[[SUICIDES_NO, POPULATION]].sum()
    data_by_year["per_capita"] = data_by_year[SUICIDES_NO] / data_by_year[POPULATION]
    plt.plot(data_by_year.index.values, data_by_year["per_capita"])
    plt.xlabel("Years")
    plt.ylabel("Suicide Rate")
    plt.show()


def gender_suicides(data: DataFrame):
    by_sex = data.groupby(data["sex"])[[SUICIDES_NO, POPULATION]].sum()
    by_sex["total"] = by_sex[POPULATION].sum()
    by_sex["total_rate"] = by_sex[SUICIDES_NO] / by_sex["total"]

    plt.pie(by_sex["total_rate"], autopct='%.1f%%', radius=1.2, labels=["Female", "Male"])
    plt.show()


loaded_data_frame = load_csv_data()

# Is the suicide rate more prominent in some age categories than others?
age_based_analysis(loaded_data_frame)

# Which countries have the most and the least number of suicides?
by_country = country_based_analysis(loaded_data_frame)
print(by_country)

# What is the effect of the population on suicide rates?
correlation = population_size_to_suicide_rate_correlation(loaded_data_frame)
print(f"Population size correlation: {correlation}")

# What is the effect of the GDP of a country on suicide rates?
gdp_correlation = gdp_to_suicide(loaded_data_frame)
print(f"Gdp per capita correlation: {gdp_correlation}")

# What is the trend of suicide rates across all the years?
trend_across_years(loaded_data_frame)

# Is there a difference between the suicide rates of men and women?
gender_suicides(loaded_data_frame)
