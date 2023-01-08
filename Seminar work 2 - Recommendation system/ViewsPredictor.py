from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender


class ViewsPredictor:
    def __init__(self) -> None:
        """Constructs a new ViewsPredictor object, which predicts values based on number of ratings."""
        pass

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.uim = {k: 0 for k in uim.df["movieID"]}
        for k in self.uim.keys():
            self.uim[k] = uim.df[uim.df["movieID"] == k].shape[0]

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
    ap = ViewsPredictor()
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))

# Results:
#
# Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 1576
# Film: The Lord of the Rings: The Two Towers, ocena: 1528
# Film: The Lord of the Rings: The Return of the King, ocena: 1457
# Film: The Silence of the Lambs, ocena: 1431
# Film: Shrek, ocena: 1404
