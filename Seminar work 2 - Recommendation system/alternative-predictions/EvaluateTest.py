from SlopeOnePredictor import SlopeOnePredictor
from Recommender import Recommender
from UserItemData import UserItemData
from MovieData import MovieData

if __name__ == "__main__":
    # Test evaluate.
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat',
                       min_ratings=1000, to_date='1.1.2008')
    rp = SlopeOnePredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    uim_test = UserItemData('data/user_ratedmovies.dat',
                            min_ratings=200, from_date='2.1.2008')

    print("Started calculations")

    mse, mae, precision, recall, f = rec.evaluate(uim_test, 20)
    print(mse, mae, precision, recall, f)
