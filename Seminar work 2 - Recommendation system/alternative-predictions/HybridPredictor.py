from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
from SlopeOnePredictor import SlopeOnePredictor
from ItemBasedPredictor import ItemBasedPredictor
from ViewsPredictor import ViewsPredictor


class HybridPredictor:
    def __init__(self):
        """
        Constructs a new HybridPredictor object that predicts by averaging
        the ratings predicted by other predictors.
        """
        self.views_predictor = ViewsPredictor()
        self.item_based_predictor = ItemBasedPredictor()
        self.slope_one_predictor = SlopeOnePredictor()

    def fit(self, uim):
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.views_predictor.fit(uim)
        self.item_based_predictor.fit(uim)
        self.slope_one_predictor.fit(uim)

    def predict(self, user_id):
        """
        Predict values by combining multiple predictors.
        As we are predicting a numerical output, we can average the
        predictions given by the different predictors.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        vp = self.views_predictor.predict(user_id)
        ibp = self.item_based_predictor.predict(user_id)
        sop = self.slope_one_predictor.predict(user_id)

        # Normalize on a scale of 1 to 5
        def normalize(val):
            if val == max(vp.values()):
                return 5
            if val == min(vp.values()):
                return 1
            return (val - min(vp.values())) * (5 - 1) / (max(vp.values()) - min(vp.values())) + 1

        vp = {k: normalize(v) for k, v in vp.items()}
        items = {k: (vp[k] + ibp[k] + sop[k]) / 3 for k in vp.keys()}
        return items


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    hp = HybridPredictor()
    rec = Recommender(hp)
    rec.fit(uim)

    print("Predictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
