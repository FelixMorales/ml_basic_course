from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
from sklearn import metrics

class LogisticRegressionModel(BaseModel):

    def __init__(self, C=0.1, random_state=1):
        BaseModel.__init__(
            self, 
            model=LogisticRegression(C=C, random_state=random_state, solver='liblinear', class_weight='balanced'), 
            model_name='Logistic Regression'
        )

    def train(self, X_train, y_train, X_test, y_test):
        C_start = 0.1
        C_end = 5
        C_inc = 0.1

        C_values, recall_scores = [], []

        C_val = C_start
        best_recall_score = 0
        i = 0
        while (C_val < C_end):
            C_values.append(C_val)
            self.model = LogisticRegression(C=C_val, random_state=42, solver='liblinear', class_weight='balanced')
            self.model.fit(X_train, y_train.ravel())
            self.predict(X_test)
            recall_score = metrics.recall_score(y_test, self.y_predict)
            recall_scores.append(recall_score)
            if (recall_score > best_recall_score):
                best_recall_score = recall_score
                
            C_val = C_val + C_inc
            i += 1
            

        best_score_C_val = C_values[recall_scores.index(best_recall_score)]
        print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))
