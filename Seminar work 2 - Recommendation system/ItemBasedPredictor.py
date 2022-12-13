from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender


class ItemBasedPredictor:
    def __init__(self, min_values, threshold):
        pass

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.similarities = dict()

        self.uim = {k: 0 for k in uim.df["movieID"]}
        for k in self.uim.keys():
            if uim.df[uim.df["movieID"] == k]["rating"].shape[0] > self.n:
                self.uim[k] = uim.df[uim.df["movieID"] == k]["rating"].std()

    def predict(self, user_id: int) -> dict[int, int]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        return self.uim.copy()

    def similarity(self, p1, p2):
        pass


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    print(uim.movies)
    print("Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716): ",
          rp.similarity(1580, 2716))
    print("Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527): ",
          rp.similarity(1580, 527))
    print("Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780): ",
          rp.similarity(1580, 780))

    print("Predictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
