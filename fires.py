import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os

np.random.seed(42)

from flask import Flask, render_template, redirect, url_for
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string2'

from flask_bootstrap import Bootstrap
Bootstrap(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class LabForm(FlaskForm):
    latitude = StringField('latitude', validators=[DataRequired()])
    longitude = StringField('longitude', validators=[DataRequired()])
    month = StringField('month', validators=[DataRequired()])
    day = StringField('day', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[float(form.latitude.data),
                            float(form.longitude.data),
                            str(form.month.data),
                            str(form.day.data),
                            float(form.avg_temp.data),
                            float(form.max_temp.data),
                            float(form.max_wind_speed.data),
                            float(form.avg_wind.data)]])
        print(X_test.shape)
        print(X_test)
        fires = pd.read_csv('./sanbul-5.csv', sep=',')
        fires_num = fires.drop(["burned_area"], axis=1)
        num = fires_num.drop(['month', 'day'], axis=1)

        #DataFrame으로 컬럼을 추가해주고
        df = pd.DataFrame(X_test, columns=["latitude", "longitude", "month", "day", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"])
        #df_num = df.drop(["month", "day"], axis=1)
        print(df)

        # pipeline
        num_pipeline = Pipeline([('std_scaler', StandardScaler()), ])

        #df_num_tr = num_pipeline.fit(df_num)
        #print(df_num_tr)

#        fires_num_tr = num_pipeline.fit(fires_num)  # 문제시 여기 수정해보자.
        #print(fires_num_tr)

        num_attribs = list(num)
        cat_attribs = ["month", "day"]

        full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs), ])


        fires_prepared = full_pipeline.fit_transform(fires)
        X_test = full_pipeline.fit_transform(df)

#################
        # X = fires.values[:, 0:8]
        # y = fires.values[:, 8]
        #
        # scaler = MinMaxScaler()
        # scaler.fit(X)

        MODEL_NAME = "my_pima_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Taeho\\Desktop\\대학교 자료\\20년 4학년 1학기\\인공지능\\Term_Project\\소스코드\\my-first-project-279810-aca50529313d.json"
        project_id = "my-first-project-279810"
        model_id = MODEL_NAME
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/versions/v0001/"
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        input_data_json = {"signature_name": "serving_default", "instances": X_test.tolist()}

        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()
        if "error" in response:
            raise RuntimeError(response["error"])

        predD = np.array([pred['dense_2'] for pred in response["predictions"]])
        print(predD[0][0])
        res = predD[0][0]

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)


if __name__ == '__main__':
    app.run()