import pandas as pd
from datetime import datetime


class MovieData:
    def __init__(self, path: str) -> None:
        """
        Constructs a new UserItemData object which contains the dataframe.

        :param path: Path to the data file.
        """
        self.path = path
        self.df = pd.read_table(path, encoding_errors="ignore")

    def get_title(self, movie_id: int) -> str:
        """
        Returns the movie title based on the movieID.

        :param movieID: The movie ID.
        :returns: The movie title.
        """
        return self.df[self.df["id"] == movie_id]["title"].to_list()[0]


if __name__ == "__main__":
    md = MovieData("data/movies.dat")
    print(md.get_title(1))
