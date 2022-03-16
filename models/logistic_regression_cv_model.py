from sklearn.linear_model import LogisticRegressionCV
from models.base_model import BaseModel

class LogisticRegressionCVModel(BaseModel):
    
    def __init__(self, n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight='balanced'):
        BaseModel.__init__(
            self, 
            model=LogisticRegressionCV(n_jobs=n_jobs, random_state=random_state, Cs=Cs, cv=cv, refit=refit, class_weight=class_weight, solver='liblinear'), 
            model_name='Logistic Regression CV'
        )