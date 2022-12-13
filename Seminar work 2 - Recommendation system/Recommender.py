from UserItemData import UserItemData
from MovieData import MovieData
from RandomPredictor import RandomPredictor


class Recommender:
    def __init__(self, predictor: RandomPredictor) -> None:
        """
        Constructs a new Recommender object that recommends options based on the given predictor.

        :param predictor: The predictor.
        """
        self.predictor = predictor

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.uim = uim
        self.predictor.fit(uim)

    def recommend(self, user_id: int = 1, n: int = 10, rec_seen: bool = False) -> list[int, int | float]:
        """
        Recommends the data based on the predictor.

        :param user_id: The user id.
        :param n: The number of predictions.
        :param rec_seen: Signifies if the recommender should recommend already seen movies.
        :returns: The list of movie ids and ratings.
        """
        pred = self.predictor.predict(user_id)
        seen_movies = set(
            self.uim.df[self.uim.df["userID"] == user_id]["movieID"])
        if not rec_seen:
            pred = {k: v for k, v in pred.items() if k not in seen_movies}
        return [(k, v) for k, v in sorted(pred.items(), key=lambda item: item[1], reverse=True)][0:n]


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
