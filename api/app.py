from flask  import Flask , jsonify , request 
from handle import prediction 



app = Flask(__name__) 

@app.post('/predict')

def predict(): 
  data = request.json
  try : 
    test = data['text']
  except KeyError : 
    return  jsonify({"error":"No text was sent"})

  test = [test]
  pred = prediction(test) 
  try : 
    result = jsonify(pred[0])
  except TypeError as e : 
    return jsonify({"error ":str(e)})
  
  return result 


if  __name__ == '__main__' : 
    app.run(host='0.0.0.0', debug=True )