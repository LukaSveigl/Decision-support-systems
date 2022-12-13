import pandas as pd
from datetime import datetime


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
        limit_date = datetime.strptime(from_date, format_data)
        indices = []
        for i in range(self.df.shape[0]):
            row_date = datetime.strptime(".".join(
                [str(self.df["date_day"][i]), str(self.df["date_month"][i]), str(self.df["date_year"][i])]), format_data)
            if row_date >= limit_date:
                indices.append(i)
        self.df = self.df.loc[self.df.index[indices]]
        self.df.reset_index(inplace=True)

    def _limit_to_date(self, to_date: str) -> None:
        """
        Filters the dataframe based on the to_date parameter.

        :param to_date: The upper limit for date filtering.
        """
        format_data = "%d.%m.%Y"
        limit_date = datetime.strptime(to_date, format_data)
        indices = []
        for i in range(self.df.shape[0]):
            row_date = datetime.strptime(".".join(
                [str(self.df["date_day"][i]), str(self.df["date_month"][i]), str(self.df["date_year"][i])]), format_data)
            if row_date <= limit_date:
                indices.append(i)
        self.df = self.df.loc[self.df.index[indices]]
        self.df.reset_index(inplace=True)

    def _limit_ratings(self, min_ratings: int) -> None:
        """
        Filters the dataframe based on the min_ratings parameter.

        :param to_date: The limit for how many ratings a movie can have.
        """
        dict_movies = dict()
        for i in range(self.df.shape[0]):
            if self.df["movieID"][i] in dict_movies:
                dict_movies[self.df["movieID"][i]] += 1
            else:
                dict_movies[self.df["movieID"][i]] = 1
        self.df = self.df[self.df["movieID"].isin(
            {k: v for k, v in dict_movies.items() if v >= min_ratings}.keys())]

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
