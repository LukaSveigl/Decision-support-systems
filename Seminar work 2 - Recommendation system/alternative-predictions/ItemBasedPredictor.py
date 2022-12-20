from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
import pandas as pd
import numpy as np
import math


class ItemBasedPredictor:
    def __init__(self, min_values: int = 0, threshold: int = 0) -> None:
        """
        Constructs a new ItemBasedPredictor object that predicts ratings based similarities between items.
        """
        self.min_values = min_values
        self.threshold = threshold

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.uim = uim
        self.df = uim.df.copy()
        self.df.drop(columns=["date_day", "date_month", "date_year",
                     "date_hour", "date_minute", "date_second"], inplace=True)

        self.average_ratings = self.df.groupby("userID")["rating"].mean()
        self.df["rating"] = (self.df.set_index(
            ["userID"])["rating"] - self.average_ratings).values

        movie_movie = pd.DataFrame(
            columns=self.df["movieID"].unique(), index=self.df["movieID"].unique())

        npmm = movie_movie.fillna(0).to_numpy()
        npmm = npmm.astype("float64")
        items = len(npmm)
        for i in range(items):
            for j in range(items):
                if i == j:
                    continue

                # If value already calculated, skip.
                if npmm[i][j] != 0.0:
                    continue

                mid1 = self.df["movieID"].unique()[i]
                mid2 = self.df["movieID"].unique()[j]

                common_users = set(self.df[self.df["movieID"] == mid1]["userID"].values).intersection(
                    set(self.df[self.df["movieID"] == mid2]["userID"].values))
                if len(common_users) == 0 or len(common_users) < self.min_values:
                    continue

                similarity = np.sum(self.df[(self.df["movieID"] == mid1) & (
                    self.df["userID"].isin(common_users))]["rating"].values * self.df[(self.df["movieID"] == mid2) & (
                        self.df["userID"].isin(common_users))]["rating"].values)

                fis = np.sum(np.power(self.df[(self.df["movieID"] == mid1) & (
                    self.df["userID"].isin(common_users))]["rating"].values, 2))

                sis = np.sum(np.power(self.df[(self.df["movieID"] == mid2) & (
                    self.df["userID"].isin(common_users))]["rating"].values, 2))

                roots = math.sqrt(fis) * math.sqrt(sis)
                if roots == 0:
                    npmm[i, j] = 0.0
                    npmm[j, i] = 0.0
                    continue
                else:
                    if similarity / roots < self.threshold:
                        npmm[i, j] = 0.0
                        npmm[j, i] = 0.0
                    else:
                        # Set values for both directions, no need to calculate twice.
                        npmm[i, j] = similarity / roots
                        npmm[j, i] = similarity / roots
        self.df = pd.DataFrame(
            npmm, columns=self.df["movieID"].unique(), index=self.df["movieID"].unique())

    def predict(self, user_id: int) -> dict[int, int | float]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        items = {k: 0 for k in self.uim.df["movieID"]}
        for k in items.keys():
            prediction = 0
            divisor = 0
            for k1 in items.keys():
                if k != k1 and len(self.uim.df[(self.uim.df["movieID"] == k1) & (self.uim.df["userID"] == user_id)]) != 0:
                    similarity = self.df.loc[k, k1]
                    prediction += similarity * self.uim.df[(self.uim.df["movieID"] == k1) & (
                        self.uim.df["userID"] == user_id)]["rating"].values[0]
                    divisor += similarity
            if divisor != 0:
                prediction = (
                    self.average_ratings[user_id] + (prediction / divisor)) / 2
            else:
                prediction = self.average_ratings[user_id]
            items[k] = prediction
        return items.copy()

    def similarity(self, p1: int, p2: int) -> int:
        """
        Finds the similarity between the given movies.

        :param p1: The first movie id.
        :param p2: The second movie id.
        """
        return self.df.loc[p1, p2]

    def similar_items(self, item: int, n: int) -> list[int, int]:
        """
        Finds n most similar movies to item.

        :param item: The movie.
        :param n: How many movies should be selected.
        """
        row = pd.DataFrame(self.df.loc[item, :])
        movies = row.sort_values(
            item, axis=0, ascending=False).iloc[:n, :].T.columns
        similarities = row.sort_values(
            item, axis=0, ascending=False).iloc[:n, :].T.values[0]

        return zip(movies, similarities)


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    # print(uim.df)
    print("Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716): ",
          rp.similarity(1580, 2716))
    print("Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527): ",
          rp.similarity(1580, 527))
    print("Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780): ",
          rp.similarity(1580, 780))

    # Predictions.
    print("\nPredictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))

    print("\n20 most similar pairs: ")
    # Find highest for each movie.
    movies = dict.fromkeys(rp.df.columns)
    for movie in movies.keys():
        # Find max for each movie but exclude itself.
        similarity = rp.df.loc[rp.df.index != movie, movie].max()
        similar_movie = rp.df.loc[rp.df.index != movie, movie].idxmax()
        movies[movie] = (similarity, similar_movie)

    for movie, similarity in sorted(movies.items(), key=lambda item: item[1][0], reverse=True)[0:20]:
        print("Movie \"{}\" is similar to \"{}\", similarity: {}".format(
              md.get_title(movie), md.get_title(similarity[1]), similarity[0]))

    # Similar items.
    rec_items = rp.similar_items(4993, 10)
    print('\nFilmi podobni "The Lord of the Rings: The Fellowship of the Ring": ')
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
