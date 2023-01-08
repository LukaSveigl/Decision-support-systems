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
        self.uim = {k: 0 for k in uim.df["isbn"]}
        for k in self.uim.keys():
            self.uim[k] = uim.df[uim.df["isbn"] == k].shape[0]

    def predict(self, user_id: int) -> dict[int, int]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        return self.uim.copy()


if __name__ == "__main__":
    md = MovieData('alternative-predictions/data/BX_Books.csv')
    uim = UserItemData(
        'alternative-predictions/data/Preprocessed_data.csv', min_ratings=500)
    ap = ViewsPredictor()
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=153662, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Knjiga: {}, ocena: {}".format(md.get_title(idmovie), val))

# Results:
#
# Knjiga: The Da Vinci Code, ocena: 883
# Knjiga: Angels & Demons, ocena: 586
# Knjiga: The Firm, ocena: 529
