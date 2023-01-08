from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender


class AveragePredictor:
    def __init__(self, b: int = 0) -> None:
        """
        Constructs a new AveragePredictor object that predicts ratings based on average ratings.

        :param b: The average formula parameter.
        """
        self.b = b
        if b < 0:
            raise ValueError(b + " must be higher than or equal to 0")

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.uim = {k: 0 for k in uim.df["movieID"]}
        for k in self.uim.keys():
            vs = sum(uim.df[uim.df["movieID"] == k]["rating"])
            g_avg = uim.df["rating"].sum() / uim.df.shape[0]
            n = uim.df[uim.df["movieID"] == k].shape[0]
            self.uim[k] = (vs + self.b * g_avg) / (n + self.b)

    def predict(self, user_id: int) -> dict[int, int | float]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        return self.uim.copy()


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = AveragePredictor(100)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))

# Results:
#
# Film: The Usual Suspects, ocena: 4.225944245560473
# Film: The Godfather: Part II, ocena: 4.146907937910189
# Film: Cidade de Deus, ocena: 4.116538340205236
# Film: The Dark Knight, ocena: 4.10413904093503
# Film: 12 Angry Men, ocena: 4.103639627096175
