import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from models.naive_bayes_model import NaiveBayesModel
from models.random_forst_model import RandomForestModel
from models.logistic_regression_model import LogisticRegressionModel
from models.logistic_regression_cv_model import LogisticRegressionCV, LogisticRegressionCVModel

## Load and review data

df = pd.read_csv('./data/pima-data.csv')

## Delete correlated feature
del df['skin']

## Check correlation
utils.plot_corr(df)

## Check data types (all feature need to be numeric type)

# map boolean to integer
df['diabetes'] = df['diabetes'].map(utils.map_boolean_to_integer())
utils.print_data(df.head(5), 'Boolean to Integer')

# check true/false ration
utils.check_boolean_ration(df)

## Split the data (70% training, 30% test)

featured_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_names = ['diabetes']

X = df[featured_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

utils.verify_split_data(df, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

## Post split processing data

# manage missing information
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

## Using Naive Bayes Model
nb_model = NaiveBayesModel()
nb_model.train(X_train, y_train)

nb_model.predict(x=X_test)
utils.show_accuracy(y=y_test, y_predicted=nb_model.y_predict, 
                        data_set_name='Test', model_name=nb_model.name)

utils.show_metrics(y_test, nb_model.y_predict, model_name=nb_model.name)


## Using Random Forest Model
rf_model = RandomForestModel(random_state=42)
rf_model.train(X_train, y_train)

rf_model.predict(x=X_test)

utils.show_metrics(y_test, rf_model.y_predict, model_name=rf_model.name)

## Using Logistic Regression Model

lr_model = LogisticRegressionModel(C=0.7, random_state=42)
lr_model.train(X_train, y_train, X_test, y_test)

utils.show_metrics(y_test, lr_model.y_predict, model_name=lr_model.name)

## Using Logistic Regression Model CV
lrcv_model = LogisticRegressionCVModel()
lrcv_model.train(X_train, y_train)

lrcv_model.predict(x=X_test)
utils.show_accuracy(y=y_test, y_predicted=lrcv_model.y_predict, 
                        data_set_name='Test', model_name=lrcv_model.name)

utils.show_metrics(y_test, lrcv_model.y_predict, model_name=lrcv_model.name)