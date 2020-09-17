import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"

# Common imports
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = ""
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Ignore useless warnings (see SciPy issue #5998)
import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

import os
import tarfile
import urllib
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "http://sjabber.dothome.co.kr/"
FIRES_PATH = os.path.join("C:/Users/Taeho", "Project2/Termproject")
FIRES_URL = DOWNLOAD_ROOT + "datasets/sanbul/sanbul-5.csv"


def fetch_fires_data(fires_url=FIRES_URL, fires_path=FIRES_PATH):
    if not os.path.isdir(fires_path):
        os.makedirs(fires_path)
    tgz_path = os.path.join(fires_path, "sanbul-5.csv")
    urllib.request.urlretrieve(fires_url, tgz_path)
    fires_tgz = tarfile.open(tgz_path)
    fires_tgz.extractall(path=fires_path)
    fires_tgz.close()

    fetch_fires_data()

def load_fires_data(fires_path=FIRES_PATH):
    csv_path = os.path.join(fires_path, "sanbul-5.csv")
    return pd.read_csv(csv_path)


fires = load_fires_data()

fires.head()

fires.info()

fires.describe()

fires["month"].value_counts()

fires["day"].value_counts()

fires['burned_area'] = np.log(fires['burned_area'] + 1)

print("\n\nHistogram plots:\n")
fires['burned_area'].hist(bins=50, figsize=(20, 15))
plt.title("burned_area")
save_fig("attribute_histogram_plots")
plt.show()

np.random.seed(42)

#
# from zlib import crc32
#
#
# def test_set_check(identifier, test_ratio):
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32
#
# #
# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

test_set.head()

fires["month"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nMonth category proportion: \n",
      strat_test_set["month"].value_counts() / len(strat_test_set))

print("\nOverall month category proportion: \n",
      fires["month"].value_counts() / len(fires))

###################################################################### 1-5 완료

#from pandas.plotting import scatter_matrix 위에 선언하여 코드 모양 정리.
# burned_area와 가장 상관관계가 높은 특성 몇개만 plot한다.

attributes = ["burned_area", "max_temp", "avg_temp",
              "max_wind_speed"]

scatter_matrix(fires[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
################################################################### 1-6 완료
# 지역별로 'burned_area'에 대해 plot한다. 원의반경은 max_temp(옵션 s), 컬러는 burned_area(옵션 c)를 의미한다.

fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=fires["max_temp"], label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
save_fig("Fires_Matrix")

#save_fig("longitude_latitude_scatterplot")
plt.show()

###################################################################1-7 완료
# print("\nConverted area dataset: \n", fires.head())
#
# fires_labels = fires["burned_area"].copy()
# fires_labels = fires['burned_area'].astype('float32')
# fires = fires.drop("burned_area", axis=1)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# fires['longitude'] = scaler.fit_transform(fires['longitude'].values.reshape(-1,1))
# fires['latitude'] = scaler.fit_transform(fires['latitude'].values.reshape(-1,1))
#
# fires['avg_temp'] = scaler.fit_transform(fires['avg_temp'].values.reshape(-1,1))
# fires['max_temp'] = scaler.fit_transform(fires['max_temp'].values.reshape(-1,1))
# fires['max_wind_speed'] = scaler.fit_transform(fires['max_wind_speed'].values.reshape(-1,1))
# fires['avg_wind'] = scaler.fit_transform(fires['avg_wind'].values.reshape(-1,1))
#
# fires = pd.get_dummies(fires, columns=['longitude', 'latitude', 'month', 'day'])
# print("\nScaled train dataset:\n", fires)
#
# from sklearn.model_selection import train_test_split
# X_train_full, X_test, y_train, y_test = train_test_split(fires, fires_labels, test_size=0.2, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train, test_size=0.1, random_state=42)
#
# print("\nTrain dataset:\n", X_train)
# print("\nTrain target:\n", y_train)


################################################################################################### 카테고리형 -> 더미데이터
#fires = strat_train_set.copy()

corr_matrix = fires.corr()
print(corr_matrix["burned_area"].sort_values(ascending=False))

################################################################################################### 1-8 완료

fires = strat_train_set.drop("burned_area", axis=1)  # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()

sample_incomplete_rows = fires[fires.isnull().any(axis=1)].head()
sample_incomplete_rows

fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()
fires_num = fires.drop(["month", "day"], axis=1)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
fires_cat = fires[["month"]]
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)
print("\n\nFires_cat_1hot:\n", fires_cat_1hot)
cat_encoder = OneHotEncoder(sparse=False)
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)
print("\n\nCat_month_encoder.categories_:\n", cat_encoder.categories_)

cat_encoder2 = OneHotEncoder()
fires_cat = fires[["day"]]
fires_cat2_1hot = cat_encoder2.fit_transform(fires_cat)
print("\n\nFires_cat2_1hot:\n", fires_cat2_1hot)
cat_encoder2 = OneHotEncoder(sparse=False)
fires_cat2_1hot = cat_encoder2.fit_transform(fires_cat)
print("\n\nCat_day_encoder.categories_:\n", cat_encoder2.categories_)
#################################################################################### 1-9
print("\n\n\n##################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('std_scaler', StandardScaler()), ])

fires_num_tr = num_pipeline.fit_transform(fires_num)
# 마지막을 제외한 모든 estimator는 반드시 transformer여야 한다.
# 마지막 estimator는 반드시 fit_estimator() 메소드를 가져야함.

print("\n\nFires_num_tr: \n", fires_num_tr)

from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer

num_attribs = list(fires_num)
cat_attribs = ["month", "day"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs), ])

fires_prepared = full_pipeline.fit_transform(fires)
###################################################################################10
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

###트레이닝 세트 크기 조절
n_samples, n_features = 300, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()), ('sgdregressor', SGDRegressor())])

print("\n")


##########PART 2#############
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def plot_learning_curves (model, X, y) :
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)


sgd_reg = SGDRegressor()
sgd_reg.fit(fires_prepared, fires_labels)

svm_reg = SVR()
svm_reg.fit( fires_prepared, fires_labels)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(fires_prepared, fires_labels)

dt_reg = DecisionTreeRegressor(max_depth=2, random_state=None, splitter='best')
dt_reg.fit(fires_prepared, fires_labels)

# print("sgd_reg 값 : ", sgd_reg.get_params().keys())
# print("svm_reg 값 : ", svm_reg.get_params().keys())
# print("forest_reg 값 : ", forest_reg.get_params().keys())
# print("dt_reg 값 : ", dt_reg.get_params().keys())

##########SGD
params_sgd = {'alpha': [0.1, 0.5, 0.8],
          'average': [5, 100, 7, 4],
          'early_stopping': [False],
          'epsilon':[0.1, 1.0, 1.5]}


grid_search_cv = GridSearchCV(sgd_reg, params_sgd, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)

sgd_best_model_cv = grid_search_cv.best_estimator_
print(sgd_best_model_cv)

#SGD -RMSE
fires_predictions = sgd_reg.predict(fires_prepared)
sgd_rmse = mean_squared_error(fires_labels, fires_predictions, squared=False)
sgd_rmse_reverted = (np.exp(sgd_rmse)-1)
print("\nSGD - RMSE(train set):\n", sgd_rmse_reverted)

#SGD -learning curves
plot_learning_curves(sgd_reg, X, y)
plt.axis([0, 300, 0, 3])
print("\n")
plt.ylabel("SGD(train)-RMSE", fontsize=14)
plt.show()

#SGD -cross_val_scores
from sklearn.model_selection import cross_val_score

sgd_scores = cross_val_score(sgd_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


sgd_rmse_scores = np.sqrt(-sgd_scores)
print("\nLinear Regression scores (train set): \n")
display_scores(sgd_rmse_scores)

#SGD - mean_squared_error
sgd_scores = cross_val_score(sgd_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

sgd_rmse_scores = np.sqrt(-sgd_scores)
print("\nLinear Regression scores (test set): \n")
display_scores(sgd_rmse_scores)

#########################################################
#########SVM
params_svm = {'kernel': ['linear', 'poly', 'rbf'],
              'C': [0.1, 1, 10, 100],
              'degree': [2, 3, 4],
              'epsilon': [0.1, 1.0, 1.5]}
grid_search_cv = GridSearchCV(svm_reg, params_svm, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
svm_best_model_cv = grid_search_cv.best_estimator_
print(svm_best_model_cv)

#SVM-RMSE
fires_predictions = svm_reg.predict(fires_prepared)
svm_rmse = mean_squared_error(fires_labels, fires_predictions, squared=False)
svm_rmse_reverted = (np.exp(svm_rmse) - 1)
print("\nSVM - RMSE(train set):\n", svm_rmse_reverted)

#SVM-learning curve
plot_learning_curves(svm_reg, X, y)
plt.axis([0, 300, 0, 3])
print("\n")
plt.ylabel("SVM(train)-RMSE", fontsize=14)
plt.show()

#SVM - cross_val_score
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


svm_rmse_scores = np.sqrt(-svm_scores)
print("\nLinear Regression scores (train set): \n")
display_scores(svm_rmse_scores)

#SVM - mean_squared_error
svm_scores = cross_val_score(svm_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

svm_rmse_scores = np.sqrt(-svm_scores)
print("\nLinear Regression scores (test set): \n")
display_scores(svm_rmse_scores)


#randomforest
params_rf = {'min_weight_fraction_leaf': [0, 0.5],
             'max_leaf_nodes': [4, 8, 6, 3],
             'min_impurity_decrease': [0.3, 0.4, 0.7],
             'max_depth': [1, 7, 8]}
grid_search_cv = GridSearchCV(forest_reg, params_rf, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
rf_best_model_cv = grid_search_cv.best_estimator_
print(rf_best_model_cv)

# randomforest-RMSE
fires_predictions = forest_reg.predict(fires_prepared)
rf_rmse = mean_squared_error(fires_labels, fires_predictions, squared=False)
rf_rmse_reverted = np.exp(rf_rmse) - 1
print("\nRF - RMSE(train set):\n", rf_rmse_reverted)

# randomforest-learning curve
plot_learning_curves(forest_reg, X, y)
plt.axis([0, 300, 0, 3])
print("\n")
plt.ylabel("RF(train)-RMSE", fontsize=14)
plt.show()

#RF(RandomForest) -cross_val_scores
from sklearn.model_selection import cross_val_score

RF_scores = cross_val_score(forest_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


RF_rmse_scores = np.sqrt(-RF_scores)
print("\nLinear Regression scores (train set): \n")
display_scores(RF_rmse_scores)

#RF(randomForest) - mean_squared_error
rf_scores = cross_val_score(forest_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

rf_rmse_scores = np.sqrt(-rf_scores)
print("\nLinear Regression scores (test set): \n")
display_scores(rf_rmse_scores)

# Decision tree
params_dt = {'ccp_alpha': [0.1, 1.0, 9.8, 4.3],
             'max_features': ["auto", "sqrt", "log2", None],
             'max_depth': [2, 3, 4, 8],
             'splitter': ["best", "random"]}
grid_search_cv = GridSearchCV(dt_reg, params_dt, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
dt_best_model_cv = grid_search_cv.best_estimator_
print(dt_best_model_cv)

# Decision tree - RMSE
fires_predictions = dt_reg.predict(fires_prepared)
dt_rmse = mean_squared_error(fires_labels, fires_predictions, squared=False)
dt_rmse_reverted = np.exp(dt_rmse) - 1
print("\nDT - RMSE(train set):\n", dt_rmse_reverted)

#Decision tree - learning curve
plot_learning_curves(dt_reg, X, y)
plt.axis([0, 300, 0, 3])
print("\n")
plt.ylabel("DT(train)-RMSE", fontsize=14)
plt.show()


#DT(Decision Tree) -cross_val_scores
from sklearn.model_selection import cross_val_score

DT_scores = cross_val_score(dt_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


DT_rmse_scores = np.sqrt(-DT_scores)
print("\nLinear Regression scores (train set): \n")
display_scores(DT_rmse_scores)

#DT(Decision Tree) - mean_squared_error
dt_scores = cross_val_score(dt_reg, fires_prepared, fires_labels,
                             scoring="neg_mean_squared_error", cv=10)

dt_rmse_scores = np.sqrt(-dt_scores)
print("\nLinear Regression scores (test set): \n")
display_scores(dt_rmse_scores)

import tensorflow as tf
from tensorflow import keras

X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test = fires_prepared
y_test = fires_labels

np.random.seed(42)
tf.random.set_seed(42)

mlp_model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

mlp_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))

history = mlp_model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))
mse_test = mlp_model.evaluate(X_test, fires_labels)
X_new = X_test[:10]
y_pred = mlp_model.predict(X_new)

history = mlp_model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))
mse_test = mlp_model.evaluate(X_test, fires_labels)
X_new = X_test[:10]
y_pred = mlp_model.predict(X_new)
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model_version = "0001"
model_name = "my_fires_model"
model_path = os.path.join( model_name, model_version)
tf.saved_model.save(mlp_model, model_path)