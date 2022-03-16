from xml.parsers.expat import model
from sklearn.naive_bayes import GaussianNB
from models.base_model import BaseModel

class NaiveBayesModel(BaseModel):

    def __init__(self):
        BaseModel.__init__(
            self, 
            model=GaussianNB(), 
            model_name='Naive Bayes'
        )