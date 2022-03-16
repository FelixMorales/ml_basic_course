from abc import ABC

class BaseModel(ABC):
    def __init__(self, model, model_name):
        self.model = model
        self.name = model_name
        self.accuracy = 0
        self.y_predict = 0
        print(f'Model: {self.model}')
        
    # x = data
    # y = labels
    def train(self, x, y):
        y = y.ravel()
        self.model.fit(x, y)
    
    def predict(self, x):
        self.y_predict = self.model.predict(x)