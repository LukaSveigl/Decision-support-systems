from UserItemData import UserItemData
from MovieData import MovieData
from RandomPredictor import RandomPredictor

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as f1
import numpy as np


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
        self.pred = self.predictor.predict(user_id)
        seen_movies = set(
            self.uim.df[self.uim.df["user_id"] == user_id]["isbn"])
        if not rec_seen:
            pred = {k: v for k, v in self.pred.items() if k not in seen_movies}
        else:
            pred = {k: v for k, v in self.pred.items()}
        return [(k, v) for k, v in sorted(pred.items(), key=lambda item: item[1], reverse=True)][0:n]

    def evaluate(self, test_data: UserItemData, n: int) -> (float):
        """
        Evaluates the predicted results agains test data.

        :param test_data: The test data.
        :param n: The number of recommended products.
        :return: The evaluation metrics (mae, rmse, recall, precision, f1)
        """
        # Calculate MAE for predictions and test data.

        # Recommend values for each user.
        """recommended = dict()
        for u in self.uim.df["userID"]:
            recommended[u] = dict(self.recommend(
                user_id=u, n=n, rec_seen=False))

        # Calculate MAE
        mae_sums = 0
        user_count = len(recommended.keys())

        averages = {k: np.mean(self.uim.df[self.uim.df["movieID"] == k]
                    ["rating"]) for k in self.uim.df["movieID"]}

        for u in recommended.keys():
            movies = {k: averages[k] for k in recommended[u].keys()}

            a = np.array([v for _, v in sorted(
                recommended[u].items(), key=lambda item: item[0])])
            b = np.array([v for _, v in sorted(
                movies.items(), key=lambda item: item[0])])

            if len(a) < 1 or len(b) < 1:
                continue

            mae_sums += np.mean(np.absolute(np.subtract(a, b)))

        mae_res = mae_sums / user_count
        # return 0, 0, 0, 0, 0

        # Calculate RMSE
        mse_sums = 0
        for u in recommended.keys():
            movies = {k: averages[k] for k in recommended[u].keys()}

            a = np.array([v for _, v in sorted(
                recommended[u].items(), key=lambda item: item[0])])
            b = np.array([v for _, v in sorted(
                movies.items(), key=lambda item: item[0])])

            if len(a) < 1 or len(b) < 1:
                continue

            mse_sums += np.mean(np.square(np.subtract(a, b)))
            # mse_sums += mse(a, b)  # np.mean(np.absolute(np.subtract(a, b)))
        mse_res = mse_sums / user_count
        # Calculate recall

        # Calculate precision

        # Calculate f1"""

        return 0, 0, 0, 0, 0


if __name__ == "__main__":
    md = MovieData('alternative-predictions/data/BX_Books.csv')
    uim = UserItemData(
        'alternative-predictions/data/Preprocessed_data.csv', min_ratings=400)
    rp = RandomPredictor(1, 5)
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=153662, n=5, rec_seen=False)
    for idbook, val in rec_items:
        print("Knjiga: {}, ocena: {}".format(md.get_title(idbook), val))
