from sklearn.ensemble import RandomForestClassifier
from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    
    def __init__(self, random_state=1):
        BaseModel.__init__(
            self, 
            model=RandomForestClassifier(random_state=random_state), 
            model_name='Random Forest Classifier'
        )