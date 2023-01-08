from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class MatrixFactorizationPredictor:
    def __init__(self) -> None:
        """
        Creates a new MatrixFactorizationPredictor object that predicts ratings based on matrix factorization.
        """
        pass

    def fit(self, uim: UserItemData) -> None:
        """
        Fits the data to the predictor.

        :param uim: The data.
        """
        self.uim = uim
        self.df = uim.df.drop(columns=[
                              "date_day", "date_month", "date_year", "date_hour", "date_minute", "date_second"])

        # Convert dataframe to matrix where rows are users, columns are movies and cells contain actual ratings.
        R_df = self.df.pivot(
            index="userID", columns="movieID", values="rating").fillna(0)
        R = R_df.to_numpy()

        self.rdf = R_df

        # Normalize the data.
        ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - ratings_mean.reshape(-1, 1)

        # Decompose the matrix. Rank 50 is arbitrary.
        # U is a matrix of users and their latent features.
        # sigma is a diagonal matrix of singular values.
        # Vt is a matrix of movies and their latent features.
        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)

        self.U = U
        self.sigma = sigma
        self.Vt = Vt

        # Predict ratings for all users and movies.
        all_predicted_ratings = np.dot(
            np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(
            all_predicted_ratings, columns=R_df.columns, index=self.df["userID"].unique())

    def predict(self, user_id: int) -> dict[int, int | float]:
        """
        Predicts the values for data.

        :param user_id: The user id.
        :returns: The dict of predictions.
        """
        sorted_predictions = self.preds_df.loc[user_id, :].sort_values(
            ascending=False)
        return {k: v for k, v in sorted_predictions.items()}

    def visualize_first_10(self) -> None:
        """
        Visualizes the first 10 factors.
        """
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        ax.set_title("Visualisation of first 10 factors")
        ax.matshow(self.preds_df.head(10), aspect="equal",
                   interpolation="nearest")
        ind_array = np.arange(0, len(self.preds_df.columns))
        ind_array2 = np.arange(0, 10)
        x, y = np.meshgrid(ind_array, ind_array2)

        for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
            ax.text(x_val, y_val,
                    round(self.preds_df.iloc[y_val, x_val], 1), va='center', ha='center', fontsize=5)
        ax.set_xticklabels([''] + self.preds_df.columns.tolist()[0::10])
        ax.set_yticklabels([''] + self.preds_df.index.tolist())
        plt.show()

    def visualize_matrix_decompose(self) -> None:
        """
        Visualizes the matrix decomposition.
        """
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # Plot self.U
        ax[0].set_title("U")
        ax[0].matshow(self.U, aspect="equal", interpolation="nearest")

        # Plot self.sigma
        ax[1].set_title("\u03A3")
        ax[1].matshow(self.sigma, aspect="equal", interpolation="nearest")

        # Plot self.Vt
        ax[2].set_title("V\u1D40")
        ax[2].matshow(self.Vt, aspect="equal", interpolation="nearest")

        plt.show()


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    mfp = MatrixFactorizationPredictor()
    rec = Recommender(mfp)
    rec.fit(uim)

    # Predictions.
    print("\nPredictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))

    # mfp.visualize_first_10()
    # mfp.visualize_matrix_decompose()

# Results:
#
# Predictions for 78:
# Film: Men in Black, ocena: 1.8204205103868067
# Film: Spider-Man, ocena: 1.5347774658228888
# Film: Toy story, ocena: 1.390759056915183
# Film: Titanic, ocena: 1.3640562754303271
# Film: Austin Powers: The Spy Who Shagged Me, ocena: 0.9681013094861783
# Film: Sin City, ocena: 0.8394283325362564
# Film: Mrs. Doubtfire, ocena: 0.836297642176647
# Film: Indiana Jones and the Last Crusade, ocena: 0.8318333017071369
# Film: Speed, ocena: 0.8045875158317601
# Film: Good Will Hunting, ocena: 0.7890629128627
# Film: The Silence of the Lambs, ocena: 0.7208267064614353
# Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 0.6847876754223341
# Film: The Fifth Element, ocena: 0.6540223951172395
# Film: Star Wars: Episode I - The Phantom Menace, ocena: 0.46675515098110365
# Film: The Incredibles, ocena: 0.417184894673861
