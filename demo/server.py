#! /usr/bin/env python3
import beaker.middleware
import bottle
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
    print("this is just a test")
    bottle.response.content_type = "application/json"
    return json.dumps({"test_backend": "test"})

bottle.run(app, host="0.0.0.0", port="8080")
