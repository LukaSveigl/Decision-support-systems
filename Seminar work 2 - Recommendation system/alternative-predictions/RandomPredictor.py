from UserItemData import UserItemData
from MovieData import MovieData
import random as rd


class RandomPredictor:
    def __init__(self, min_rating: int | float, max_rating: int | float) -> None:
        """
        Constructs a new RandomPredictor object that predicts ratings randomly.

        :param min_rating: The minimum rating the predictor can predict.
        :param max_rating: The maximum rating the predictor can predict.
        """
        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.uim = {k: 0 for k in uim.df["isbn"]}

    def predict(self, user_id: int) -> dict[int, int]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        for k in self.uim.keys():
            self.uim[k] = rd.randint(self.min_rating, self.max_rating)
        return self.uim.copy()


if __name__ == "__main__":
    md = MovieData('alternative-predictions/data/BX_Books.csv')
    uim = UserItemData('alternative-predictions/data/Preprocessed_data.csv')
    rp = RandomPredictor(1, 10)
    rp.fit(uim)
    pred = rp.predict(153662)
    print(len(pred.keys()))
    items = ["0195153448", "0002005018",
             "0060973129", "0374157065", "0393045218"]
    for item in items:
        print("Knjiga: {}, ocena: {}".format(md.get_title(item), pred[item]))

# Results:
#
# 270170
# Knjiga: Classical Mythology, ocena: 2
# Knjiga: Clara Callan, ocena: 7
# Knjiga: Decision in Normandy, ocena: 4
# Knjiga: Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It, ocena: 8
# Knjiga: The Mummies of Urumchi, ocena: 8
