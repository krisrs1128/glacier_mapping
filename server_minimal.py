import bottle
from bottle import route, run,Bottle, response
app = Bottle()
import json

@app.hook("after_request")
def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163
    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    print("after request")
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
    bottle.response.headers['Access-Control-Allow-Credentials'] = True

@app.route('/hello/', method='POST')
def hello():
   bottle.response.content_type = 'application/json'
   #data = bottle.request.json
   #data["message"]="hello world!"
   data = {"message":"hello World"}
   return json.dumps(data)

app.run(host='localhost', port=8080, debug=True)
