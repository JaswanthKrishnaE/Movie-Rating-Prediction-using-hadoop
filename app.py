from flask import Flask, request, jsonify
import pickle
import numpy as np
import joblib

app = Flask(__name__)

with open('RF_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_array = np.array(data['movie_info'])
        arr = np.array(input_array)
        td = [arr]
        td_arr = np.array(td)
        predictions = model.predict(td_arr)
        sc = joblib.load('LabelEncoder.pkl')
        value = predictions
        inverse_transformed_value = sc.inverse_transform([value])[0]
        print(inverse_transformed_value)
        return jsonify({'predictions': inverse_transformed_value.tolist()}),200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)