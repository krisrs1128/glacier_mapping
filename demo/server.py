#! /usr/bin/env python3
import beaker.middleware
import bottle
import cheroot.wsgi
import json
app = bottle.Bottle()


@app.hook("after_request")
def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163

    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    print("after request")
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


@app.post("/test")
def test():
    bottle.response.content_type = 'application/json'
    print("this is just a test")
    return json.dumps({"test": "test"})

bottle.run(app, host="localhost", port="8080")
