import pandas as pd

data = pd.read_csv(
    "alternative-predictions/data/Preprocessed_data.csv", sep=",",  encoding_errors="ignore")

data.drop(columns=['Unnamed: 0', 'location', 'age',
                   'book_author', 'publisher',
                   'img_s', 'img_m', 'img_l', 'Summary', 'Language', 'Category', 'city',
                   'state', 'country'], inplace=True)

print(data.head(10))
print(data.columns)
