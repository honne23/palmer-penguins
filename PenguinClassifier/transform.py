import pandas as pd
from functools import reduce


class Transform:

    def __init__(self):
        self.features = ['bill_length_mm', 'bill_depth_mm',
                         'flipper_length_mm', 'body_mass_g']
        self.labels = ['sex', 'species']
        self.transformations = [self.clean_null_values,
                                self.collect_subset, self.normalise]

    def clean_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop all rows that have null values
        """
        print(
            f'Portion of rows with null values: {round(1-(len(df.dropna())/ len(df)),2)}%')
        return df.dropna()

    def collect_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collect a subset of features of interest + labels from data frame
        """
        return df[self.labels + self.features]

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carry out mean-normalization before creating train-test splits
        """
        df[self.features] = (df[self.features] -
                             df[self.features].mean())/df[self.features].std()
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carry out pipeline transformations
        """
        return reduce(lambda d, f: f(d), self.transformations, df)
