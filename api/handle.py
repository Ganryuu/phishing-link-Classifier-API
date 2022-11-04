import pickle 


model  = pickle.load(open('./phishing_classifier.pkl', 'rb'))


def prediction(x): 

  return model.predict(x) 