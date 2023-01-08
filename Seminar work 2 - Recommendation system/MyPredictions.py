from ItemBasedPredictor import ItemBasedPredictor
from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
import datetime
import pandas as pd

if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)

    # Add data to uim.
    # Movies:
    # 5500  Top Secret!
    # 593   The Silence of the Lambs
    # 1704  Good Will Hunting
    # 1590	Event Horizon
    # 3081	Sleepy Hollow
    # 3142	U2: Rattle and Hum
    # 4105	The Evil Dead
    # 1974	Friday the 13th
    # 1298	Pink Floyd The Wall
    # 1347	A Nightmare on Elm Street
    # 5785	Jackass: The Movie
    # 1136	Monty Python and the Holy Grail
    # 296	Pulp Fiction
    # 356	Forrest Gump
    # 61132	Tropic Thunder
    # 31696	Constantine
    # 8965	The Polar Express

    movie_ids = {5500: 4.3,
                 593: 4.4,
                 1704: 4.4,
                 1590: 4.0,
                 3081: 3.3,
                 3142: 3.8,
                 4105: 3.7,
                 1974: 3.7,
                 1298: 3.8,
                 1347: 3.7,
                 5785: 3.2,
                 1136: 4.0,
                 296: 4.1,
                 356: 3.6,
                 61132: 3.8,
                 31696: 3.5,
                 8965: 3.5}

    for k, v in movie_ids.items():
        dt = datetime.datetime.now()
        row = {"userID": [999999], "movieID": [k], "rating": [v], "date_day": [dt.day], "date_month": [dt.month],
               "date_year": [dt.year], "date_hour": [dt.hour], "date_minute": [dt.minute], "date_second": [dt.second]}
        uim.df = pd.concat([uim.df, pd.DataFrame(row)], axis=0)

    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    # Predictions for myself.
    print("\nPredictions for myself: ")
    rec_items = rec.recommend(999999, n=10, rec_seen=False)
    for idmovie, val in rec_items:
        print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))
    pass

# Results:
#
# Predictions for myself:
# Film: Batman Begins, ocena: 4.105882352941177
# Film: Ocean's Eleven, ocena: 4.105882352941176
# Film: The Fugitive, ocena: 4.105882352941176
# Film: The Bourne Identity, ocena: 4.105882352941176
# Film: Monsters, Inc., ocena: 4.072070109345024
# Film: Die Hard, ocena: 4.064341903890307
# Film: Shichinin no samurai, ocena: 4.015034769975135
# Film: The Godfather, ocena: 3.9998236518482897
# Film: Alien, ocena: 3.9965629615528675
# Film: Memento, ocena: 3.995914773075207