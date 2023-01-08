from SlopeOnePredictor import SlopeOnePredictor
from Recommender import Recommender
from UserItemData import UserItemData
from MovieData import MovieData

if __name__ == "__main__":
    # Test evaluate.
    md = MovieData('alternative-predictions/data/BX_Books.csv')
    uim = UserItemData('alternative-predictions/data/Preprocessed_data.csv',
                       min_ratings=400)
    rp = SlopeOnePredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    uim_test = UserItemData('alternative-predictions/data/Preprocessed_data.csv',
                            min_ratings=300)

    print("Started calculations")

    mse, mae, precision, recall, f = rec.evaluate(uim_test, 5)
    print(mse, mae, precision, recall, f)
