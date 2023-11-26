from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import random


app = Flask(__name__)


def gener(data):
    # Check if 'movie_info' key is present and if it contains 'genres'
    if 'movie_info' in data and 'genres' in data['movie_info']:
        no_genre = data['movie_info'][pd.isnull(data['movie_info']['genres'])]
        i = data['movie_info'][pd.isnull(data['movie_info']['genres'])].index
        listgenre = ['Science Fiction & Fantasy', 'Drama', 'Animation', 'Animation', 'Animation', 'Action & Adventure', 'Musical & Performing Arts', 'Romance', 'Action & Adventure', 'Drama', 'Comedy', 'Animation', 'Action & Adventure', 'Horror', 'Action & Adventure', 'Mystery & Suspense', 'Science Fiction & Fantasy', 'Documentary', 'Animation']

        for g in range(0, len(no_genre)):
            data['movie_info'].loc[i[g], 'genres'] = listgenre[g]
            continue

        df = data['movie_info']['genres'].str.get_dummies(',')
        data['movie_info'] = pd.concat([data['movie_info'], df], axis=1)

    return data

def encoding(data):
    # Check if 'movie_info' key is present
    if 'movie_info' in data:
        # Extract 'movie_info' and 'genres' keys from the dictionary
        movie_info = data['movie_info']
        genres = movie_info.get('genres', None)

        if genres is not None:
            # Drop the 'genres' key from 'movie_info'
            movie_info.pop('genres')

            # Perform the remaining encoding steps
            drop_column = ["genres"]
            data.drop(drop_column, axis=1, inplace=True)

            cont_col = list(data.describe())
            cat_col = list(c for c in data.columns if c not in cont_col)

            cont_data = data.loc[:, cont_col]
            cat_data = data.loc[:, cat_col]

            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            cont_data = imputer.fit_transform(cont_data)

            imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            cat_data = imputer2.fit_transform(cat_data)

            cat_data = pd.DataFrame(cat_data)
            cat_data.columns = cat_col
            cont_data = pd.DataFrame(cont_data)
            cont_data.columns = cont_col

            oe = OrdinalEncoder(categories=[['Rotten', 'Fresh', 'Certified-Fresh']])
            cat_data['tomatometer_status'] = oe.fit_transform(cat_data[['tomatometer_status']])

            data = pd.concat([cat_data, cont_data], axis=1)
            return data

    return data


def process_input_data(data):
    input_df = pd.DataFrame([data])
    transformed_data = loaded_pipeline.transform(data)
    columns = [
        'content_rating', 'tomatometer_status', 'audience_status', 'runtime',
        'tomatometer_count', 'audience_rating', 'audience_count',
        'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count',
        'tomatometer_rotten_critics_count', ' Animation', ' Anime & Manga',
        ' Art House & International', ' Classics', ' Comedy', ' Cult Movies',
        ' Documentary', ' Drama', ' Faith & Spirituality', ' Gay & Lesbian',
        ' Horror', ' Kids & Family', ' Musical & Performing Arts',
        ' Mystery & Suspense', ' Romance', ' Science Fiction & Fantasy',
        ' Special Interest', ' Sports & Fitness', ' Television', ' Western',
        'Action & Adventure', 'Animation', 'Art House & International',
        'Classics', 'Comedy', 'Cult Movies', 'Documentary', 'Drama', 'Horror',
        'Kids & Family', 'Musical & Performing Arts', 'Mystery & Suspense',
        'Romance', 'Science Fiction & Fantasy', 'Special Interest',
        'Television', 'Western'
    ]
    transformed_data = pd.DataFrame([transformed_data])
    # return transformed_data
    empty_df = pd.DataFrame(0, index=range(1), columns=columns)
    specific_columns = transformed_data.columns
    empty_df[specific_columns] = transformed_data[specific_columns].values.ravel()
    empty_df = list(empty_df.iloc[0])

    return np.array(empty_df)


model = joblib.load('./models/RF_model.pkl')
label_encoder = joblib.load('./models/LabelEncoder.pkl')
loaded_pipeline = joblib.load("./models/processed_data_pipeline.joblib")


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        input_data = data.get('movie_info')
        print("Input data:", input_data)
        if input_data is None:
            return jsonify({'error': 'Invalid input format'}), 400
        processed_data = process_input_data(input_data)
        print("Processed data:", processed_data)
        predictions = model.predict(processed_data.reshape(1, -1))
        sc = joblib.load('./models/LabelEncoder.pkl')
        value = predictions
        inverse_transformed_value = sc.inverse_transform([value])[0]
        print("Predictions:", inverse_transformed_value)
        return jsonify({'predictions': inverse_transformed_value.tolist()}), 200
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()  
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host = '0.0.0.0')
