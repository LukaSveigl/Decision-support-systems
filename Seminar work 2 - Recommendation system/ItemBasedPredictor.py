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

# Results:
#
# Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716):  0.23395523176756633
# Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527):  0.0
# Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780):  0.42466125844687624
#
# Predictions for 78:
# Film: Shichinin no samurai, ocena: 4.1604255346899635
# Film: The Usual Suspects, ocena: 4.1598990035688015
# Film: The Silence of the Lambs, ocena: 4.150210791271142
# Film: Sin City, ocena: 4.121901722984434
# Film: Monsters, Inc., ocena: 4.0913487080066435
# Film: The Incredibles, ocena: 4.0861074311757575
# Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 4.058954193209058
# Film: Batman Begins, ocena: 4.055765042884983
# Film: Die Hard, ocena: 4.045515940651294
# Film: Rain Man, ocena: 4.018325761014159
# Film: The Lord of the Rings: The Return of the King, ocena: 3.99267686416339
# Film: A Beautiful Mind, ocena: 3.9901293845673034
# Film: Good Will Hunting, ocena: 3.987198542996294
# Film: The Lord of the Rings: The Two Towers, ocena: 3.9532962920826806
# Film: Indiana Jones and the Last Crusade, ocena: 3.8810463877243455
#
# 20 most similar pairs:
# Movie "The Lord of the Rings: The Two Towers" is similar to "The Lord of the Rings: The Return of the King", similarity: 0.8439842148481417
# Movie "The Lord of the Rings: The Return of the King" is similar to "The Lord of the Rings: The Two Towers", similarity: 0.8439842148481417
# Movie "The Lord of the Rings: The Fellowship of the Ring" is similar to "The Lord of the Rings: The Two Towers", similarity: 0.8231885401761888
# Movie "Kill Bill: Vol. 2" is similar to "Kill Bill: Vol. 2", similarity: 0.7372340224381029
# Movie "Kill Bill: Vol. 2" is similar to "Kill Bill: Vol. 2", similarity: 0.7372340224381029
# Movie "Star Wars" is similar to "Star Wars: Episode V - The Empire Strikes Back", similarity: 0.7021321132220318
# Movie "Star Wars: Episode V - The Empire Strikes Back" is similar to "Star Wars", similarity: 0.7021321132220318
# Movie "Ace Ventura: Pet Detective" is similar to "The Mask", similarity: 0.6616471778494046
# Movie "The Mask" is similar to "Ace Ventura: Pet Detective", similarity: 0.6616471778494046
# Movie "Star Wars: Episode VI - Return of the Jedi" is similar to "Star Wars: Episode V - The Empire Strikes Back", similarity: 0.5992253753778948
# Movie "Independence Day" is similar to "Star Wars: Episode I - The Phantom Menace", similarity: 0.5610426219249997
# Movie "Star Wars: Episode I - The Phantom Menace" is similar to "Independence Day", similarity: 0.5610426219249997
# Movie "Austin Powers: The Spy Who Shagged Me" is similar to "Ace Ventura: Pet Detective", similarity: 0.5546511205201551
# Movie "Speed" is similar to "Pretty Woman", similarity: 0.5452283115904596
# Movie "Pretty Woman" is similar to "Speed", similarity: 0.5452283115904596
# Movie "Mrs. Doubtfire" is similar to "The Mask", similarity: 0.5398021259282235
# Movie "The Matrix Reloaded" is similar to "Star Wars: Episode I - The Phantom Menace", similarity: 0.539553095856011
# Movie "Pulp Fiction" is similar to "Reservoir Dogs", similarity: 0.5325845218198639
# Movie "Reservoir Dogs" is similar to "Pulp Fiction", similarity: 0.5325845218198639
# Movie "The Shawshank Redemption" is similar to "The Usual Suspects", similarity: 0.517724533955058
#
# Filmi podobni "The Lord of the Rings: The Fellowship of the Ring":
# Film: The Lord of the Rings: The Two Towers, ocena: 0.8231885401761888
# Film: The Lord of the Rings: The Return of the King, ocena: 0.8079374897442496
# Film: Star Wars: Episode V - The Empire Strikes Back, ocena: 0.2396194307349645
# Film: Star Wars, ocena: 0.2196558652707407
# Film: The Matrix, ocena: 0.2151555270688023
# Film: Raiders of the Lost Ark, ocena: 0.19944276706345015
# Film: The Usual Suspects, ocena: 0.18321188451910753
# Film: Blade Runner, ocena: 0.16399681315410275
# Film: Schindler's List, ocena: 0.16105905138148702
# Film: Monty Python and the Holy Grail, ocena: 0.15780453798519137
