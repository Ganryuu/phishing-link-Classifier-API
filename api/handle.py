import pickle 


model  = pickle.load(open('api/models/phishing_classifier.pkl', 'rb'))


def prediction(x): 

  return model.predict(x) 