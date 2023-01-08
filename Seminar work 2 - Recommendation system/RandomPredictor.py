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
        self.uim = {k: 0 for k in uim.df["movieID"]}

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
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rp.fit(uim)
    pred = rp.predict(78)
    print(type(pred))
    items = [1, 3, 20, 50, 100]
    for item in items:
        print("Film: {}, ocena: {}".format(md.get_title(item), pred[item]))

# Results
#
# <class 'dict'>
# Film: Toy story, ocena: 4
# Film: Grumpy Old Men, ocena: 4
# Film: Money Train, ocena: 4
# Film: The Usual Suspects, ocena: 2
# Film: City Hall, ocena: 5