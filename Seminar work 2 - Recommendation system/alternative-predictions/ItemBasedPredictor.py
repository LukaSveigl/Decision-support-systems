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
        # self.df.drop(columns=["date_day", "date_month", "date_year",
        #             "date_hour", "date_minute", "date_second"], inplace=True)

        self.average_ratings = self.df.groupby("user_id")["rating"].mean()
        self.df["rating"] = (self.df.set_index(
            ["user_id"])["rating"] - self.average_ratings).values

        movie_movie = pd.DataFrame(
            columns=self.df["isbn"].unique(), index=self.df["isbn"].unique())

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

                mid1 = self.df["isbn"].unique()[i]
                mid2 = self.df["isbn"].unique()[j]

                common_users = set(self.df[self.df["isbn"] == mid1]["user_id"].values).intersection(
                    set(self.df[self.df["isbn"] == mid2]["user_id"].values))
                if len(common_users) == 0 or len(common_users) < self.min_values:
                    continue

                similarity = np.sum(self.df[(self.df["isbn"] == mid1) & (
                    self.df["user_id"].isin(common_users))]["rating"].values * self.df[(self.df["isbn"] == mid2) & (
                        self.df["user_id"].isin(common_users))]["rating"].values)

                fis = np.sum(np.power(self.df[(self.df["isbn"] == mid1) & (
                    self.df["user_id"].isin(common_users))]["rating"].values, 2))

                sis = np.sum(np.power(self.df[(self.df["isbn"] == mid2) & (
                    self.df["user_id"].isin(common_users))]["rating"].values, 2))

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
            npmm, columns=self.df["isbn"].unique(), index=self.df["isbn"].unique())

    def predict(self, user_id: int) -> dict[int, int | float]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        items = {k: 0 for k in self.uim.df["isbn"]}
        for k in items.keys():
            prediction = 0
            divisor = 0
            for k1 in items.keys():
                if k != k1 and len(self.uim.df[(self.uim.df["isbn"] == k1) & (self.uim.df["user_id"] == user_id)]) != 0:
                    similarity = self.df.loc[k, k1]
                    prediction += similarity * self.uim.df[(self.uim.df["isbn"] == k1) & (
                        self.uim.df["user_id"] == user_id)]["rating"].values[0]
                    divisor += similarity
            if divisor != 0:
                prediction = (
                    self.average_ratings[user_id] + (prediction / divisor))
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
    md = MovieData('alternative-predictions/data/BX_Books.csv')
    uim = UserItemData(
        'alternative-predictions/data/Preprocessed_data.csv', min_ratings=400)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    # Predictions.
    print("\nPredictions for 153662: ")
    rec_items = rec.recommend(153662, n=15, rec_seen=False)
    for idbook, val in rec_items:
        print("Knjiga: {}, ocena: {}".format(md.get_title(idbook), val))

    print("\n20 most similar pairs: ")
    # Find highest for each movie.
    books = dict.fromkeys(rp.df.columns)
    for book in books.keys():
        # Find max for each movie but exclude itself.
        similarity = rp.df.loc[rp.df.index != book, book].max()
        similar_movie = rp.df.loc[rp.df.index != book, book].idxmax()
        books[book] = (similarity, similar_movie)

    for book, similarity in sorted(books.items(), key=lambda item: item[1][0], reverse=True)[0:20]:
        print("Book \"{}\" is similar to \"{}\", similarity: {}".format(
              md.get_title(book), md.get_title(similarity[1]), similarity[0]))

# Results:
#
# Predictions for 153662:
# Knjiga: The Horse Whisperer, ocena: 6.1858072477346475
# Knjiga: The Firm, ocena: 5.297892926906104
# Knjiga: The Da Vinci Code, ocena: 5.009295335878882
# Knjiga: The Poisonwood Bible: A Novel, ocena: 4.821153420093065
# Knjiga: She's Come Undone (Oprah's Book Club), ocena: 3.963834119766654
# Knjiga: The Testament, ocena: 3.6232530146796655
# Knjiga: She's Come Undone (Oprah's Book Club (Paperback)), ocena: 3.207193317498997
# Knjiga: The Nanny Diaries: A Novel, ocena: 2.934076516132335
# Knjiga: Jurassic Park, ocena: 2.8997207658329183
# Knjiga: Angels & Demons, ocena: 2.0292407612886265
# Knjiga: A Prayer for Owen Meany, ocena: 1.8928571428571428
#
# 20 most similar pairs:
# Book "Little Altars Everywhere: A Novel" is similar to "The Nanny Diaries: A Novel", similarity: 0.46304544906230805
# Book "The Nanny Diaries: A Novel" is similar to "Little Altars Everywhere: A Novel", similarity: 0.46304544906230805
# Book "The Catcher in the Rye" is similar to "Life of Pi", similarity: 0.3966352489589352
# Book "Life of Pi" is similar to "The Catcher in the Rye", similarity: 0.3966352489589352
# Book "Timeline" is similar to "The Joy Luck Club", similarity: 0.3866294976966082
# Book "The Joy Luck Club" is similar to "Timeline", similarity: 0.3866294976966082
# Book "The Firm" is similar to "The Nanny Diaries: A Novel", similarity: 0.3626614344317186
# Book "Jurassic Park" is similar to "Life of Pi", similarity: 0.3494117809775428
# Book "The Runaway Jury" is similar to "Life of Pi", similarity: 0.34247582037004926
# Book "She's Come Undone (Oprah's Book Club (Paperback))" is similar to "A Prayer for Owen Meany", similarity: 0.3257235267990078
# Book "A Prayer for Owen Meany" is similar to "She's Come Undone (Oprah's Book Club (Paperback))", similarity: 0.3257235267990078
# Book "The Five People You Meet in Heaven" is similar to "A Painted House", similarity: 0.3220240613099573
# Book "A Painted House" is similar to "The Five People You Meet in Heaven", similarity: 0.3220240613099573
# Book "Good in Bed" is similar to "The Catcher in the Rye", similarity: 0.3178231561463235
# Book "The Poisonwood Bible: A Novel" is similar to "The Catcher in the Rye", similarity: 0.3096790142153704
# Book "The No. 1 Ladies' Detective Agency (Today Show Book Club #8)" is similar to "Timeline", similarity: 0.3059778613183301
# Book "The Summons" is similar to "The Five People You Meet in Heaven", similarity: 0.29714819596081776
# Book "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))" is similar to "The Poisonwood Bible: A Novel", similarity: 0.2970368770273955
# Book "Interview with the Vampire" is similar to "The Nanny Diaries: A Novel", similarity: 0.29359942753633855
# Book "The Pelican Brief" is similar to "The Catcher in the Rye", similarity: 0.28548035828631096
