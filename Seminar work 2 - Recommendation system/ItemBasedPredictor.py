from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
from scipy.spatial.distance import pdist, squareform
import pandas as pd


class ItemBasedPredictor:
    def __init__(self, min_values: int = 0, threshold: int = 0) -> None:
        """
        Constructs a new ItemBasedPredictor object that predicts ratings based similarities between items.
        """
        self.min_values = min_values
        self.threshold = threshold
        pass

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.df = uim.df.groupby(
            ["movieID", "userID"]).size().unstack().fillna(0)

        for col in self.df.columns:
            indices = uim.df[uim.df["userID"] == col]["movieID"].tolist()
            self.df.loc[indices, col] = [
                x for x in uim.df[uim.df["userID"] == col]["rating"]]

        #
        df_mean = self.df.mean(axis=1)
        self.df = self.df - df_mean[:, None]

        self.df = pd.DataFrame(
            squareform(pdist(self.df, "cosine")),
            columns=self.df.index, index=self.df.index)
        # self.df = pd.DataFrame(
        #     squareform(pdist(self.df, "cosine")),
        #     columns=self.df.index, index=self.df.index)

        # Remove columns that have similarity under threshold.
        self.df = self.df[self.df.columns].where(
            self.df[self.df.columns] >= self.threshold, other=0)

        # Remove columns that don't have enough ratings.
        dict_movies = dict()
        for movie in uim.df["movieID"]:
            if movie in dict_movies:
                dict_movies[movie] += 1
            else:
                dict_movies[movie] = 1

        for k in dict_movies.keys():
            if dict_movies[k] < self.min_values:
                # Set entire column to 0.
                self.df[k] = 0
                # Set entire row to 0.
                self.df.loc[k, :] = 0
        self.uim = uim

    def predict(self, user_id: int) -> dict[int, int | float]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        # Find all user movies.
        user_movies = self.uim.df[self.uim.df["userID"]
                                  == user_id]["movieID"].to_list()
        # For each movie, find max similarity movie in seen movies.
        items = {k: 0 for k in self.uim.df["movieID"]}
        for k in items.keys():
            similarity = 0
            similar_movie = 0
            for um in user_movies:
                if self.df.loc[um, k] > similarity:
                    similarity = self.df.loc[um, k]
                    similar_movie = um

            user_movs = self.uim.df[self.uim.df["userID"] == user_id]
            movie_rating = user_movs[user_movs["movieID"]
                                     == similar_movie]["rating"].to_list()[0]
            # Calculate score based on similarity and score.
            items[k] = similarity * movie_rating
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

    for movie, similarity in sorted(movies.items(), key=lambda item: item[1][1], reverse=True)[0:20]:
        print("Movie \"{}\" is similar to \"{}\", similarity: {}".format(
              md.get_title(movie), md.get_title(similarity[1]), similarity[0]))

        # Similar items.
    rec_items = rp.similar_items(4993, 10)
    print('\nFilmi podobni "The Lord of the Rings: The Fellowship of the Ring": ')
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
