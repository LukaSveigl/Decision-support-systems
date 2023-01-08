import pandas as pd


class UserItemData:
    def __init__(self, path: str, from_date: str = None, to_date: str = None, min_ratings: int = None) -> None:
        """
        Constructs a new UserItemData object which contains the dataframe. The 
        dataframe can possibly be filtered based on the passed parameters.

        :param path: Path to the data file.
        :param from_date: The lower limit for date filtering.
        :param to_date: The upper limit for date filtering.
        :min_ratings: The limit for how many ratings a movie can have.
        """
        self.path = path

        self.df = pd.read_table(path, encoding_errors="ignore")

        if from_date is not None:
            self._limit_from_date(from_date)
        if to_date is not None:
            self._limit_to_date(to_date)
        if min_ratings is not None:
            self._limit_ratings(min_ratings)

    def _limit_from_date(self, from_date: str) -> None:
        """
        Filters the dataframe based on the from_date parameter.

        :param from_date: The lower limit for date filtering.
        """
        format_data = "%d.%m.%Y"
        self.df.rename(columns={"date_year": "year",
                       "date_month": "month", "date_day": "day"}, inplace=True)
        self.df["date"] = pd.to_datetime(
            self.df[["day", "month", "year"]])
        self.df = self.df[self.df["date"] >=
                          pd.to_datetime(from_date, format=format_data)]
        self.df.drop("date", axis=1)

    def _limit_to_date(self, to_date: str) -> None:
        """
        Filters the dataframe based on the to_date parameter.

        :param to_date: The upper limit for date filtering.
        """
        format_data = "%d.%m.%Y"
        self.df.rename(columns={"date_year": "year",
                       "date_month": "month", "date_day": "day"}, inplace=True)
        self.df["date"] = pd.to_datetime(
            self.df[["day", "month", "year"]])
        self.df = self.df[self.df["date"] <
                          pd.to_datetime(to_date, format=format_data)]
        self.df.drop("date", axis=1)

    def _limit_ratings(self, min_ratings: int) -> None:
        """
        Filters the dataframe based on the min_ratings parameter.

        :param to_date: The limit for how many ratings a movie can have.
        """
        counts = self.df["movieID"].value_counts()
        self.df = self.df[self.df["movieID"].isin(
            counts[counts >= min_ratings].index)]

    def read_ratings(self) -> int:
        """
        Returns how many ratings are in the dataframe.

        :returns: The number of ratings.
        """
        return self.df.shape[0]


if __name__ == "__main__":
    uim = UserItemData("data/user_ratedmovies.dat")
    print(uim.read_ratings())

    uim = UserItemData("data/user_ratedmovies.dat",
                       from_date="12.1.2007", to_date="16.2.2008", min_ratings=100)
    print(uim.read_ratings())

# Results:
#
# 855598
# 73584
