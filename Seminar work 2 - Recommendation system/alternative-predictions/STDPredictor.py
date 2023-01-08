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
        self.uim = {k: 0 for k in uim.df["isbn"]}
        for k in self.uim.keys():
            if uim.df[uim.df["isbn"] == k]["rating"].shape[0] > self.n:
                self.uim[k] = uim.df[uim.df["isbn"] == k]["rating"].std()

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
    ap = STDPredictor(100)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=153662, n=5, rec_seen=False)
    for idbook, val in rec_items:
        print("Knjiga: {}, ocena: {}".format(md.get_title(idbook), val))

# Results:
#
# Knjiga: The Da Vinci Code, ocena: 4.376584616254901
# Knjiga: Angels & Demons, ocena: 4.182695518980329
# Knjiga: The Firm, ocena: 3.973628843304148
