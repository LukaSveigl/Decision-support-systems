import pandas as pd


class UserItemData:
    def __init__(self, path: str, min_ratings: int = None) -> None:
        """
        Constructs a new UserItemData object which contains the dataframe. The 
        dataframe can possibly be filtered based on the passed parameters.

        :param path: Path to the data file.
        :min_ratings: The limit for how many ratings a movie can have.
        """
        self.path = path

        self.df = pd.read_csv(path, sep=",")
        self.df.drop(columns=['Unnamed: 0', 'location', 'age',
                              'book_author', 'publisher', "book_title",
                              'img_s', 'img_m', 'img_l', 'Summary', 'Language', 'Category', 'city',
                              'state', 'country'], inplace=True)

        if min_ratings is not None:
            self._limit_ratings(min_ratings)

    def _limit_ratings(self, min_ratings: int) -> None:
        """
        Filters the dataframe based on the min_ratings parameter.

        :param to_date: The limit for how many ratings a movie can have.
        """
        counts = self.df["isbn"].value_counts()
        self.df = self.df[self.df["isbn"].isin(
            counts[counts >= min_ratings].index)]

    def read_ratings(self) -> int:
        """
        Returns how many ratings are in the dataframe.

        :returns: The number of ratings.
        """
        return self.df.shape[0]


if __name__ == "__main__":
    uim = UserItemData("alternative-predictions/data/Preprocessed_data.csv")
    print(uim.read_ratings())

    uim = UserItemData(
        "alternative-predictions/data/Preprocessed_data.csv", min_ratings=500)
    print(uim.read_ratings())
