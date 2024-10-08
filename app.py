from flask import Flask, request, render_template

from src.pipelines.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('index.html', results=results[0])
    
if __name__ == '__main__':
    app.run('127.0.0.1')