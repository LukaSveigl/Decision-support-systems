import pandas as pd
from datetime import datetime


class MovieData:
    def __init__(self, path: str) -> None:
        """
        Constructs a new MovieData object which contains the dataframe.

        :param path: Path to the data file.
        """
        self.path = path
        self.df = pd.read_csv(path, sep=";", encoding_errors="ignore")

    def get_title(self, isbn: str) -> str:
        """
        Returns the movie title based on the movieID.

        :param movieID: The movie ID.
        :returns: The movie title.
        """
        return self.df[self.df["ISBN"] == isbn]["Book-Title"].to_list()[0]


if __name__ == "__main__":
    md = MovieData("alternative-predictions/data/BX_Books.csv")
    print(md.get_title("0743203763"))

# Results:
#
# As Hogan Said . . . : The 389 Best Things Anyone Said about How to Play Golf
