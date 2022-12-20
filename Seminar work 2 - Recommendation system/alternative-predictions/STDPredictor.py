from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender


class STDPredictor:
    def __init__(self, n: int) -> None:
        """
        Constructs a new STDPredictor object that predicts controversial ratings.

        :param n: The minimum amount of ratings of movie.
        """
        self.n = n

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
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


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = STDPredictor(100)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
