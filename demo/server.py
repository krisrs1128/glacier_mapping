#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110,E1101
import beaker.middleware
import bottle
import cheroot.wsgi
app = bottle.Bottle()

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

@app.route('/', method = 'OPTIONS')
@app.route('/<path:path>', method = 'OPTIONS')
def options_handler(path = None):
    return

@app.hook("after_request")
def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163

    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    print("after request")
    bottle.response.headers['Access-Control-Allow-Origin'] = 'http://52.247.203.144:4040'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


@app.post("/test")
def test():
    bottle.response.content_type = 'application/json'
    data = bottle.response.json
    print("this is just a test")
    bottle.response.status = 200
    return json.dumps(data)

bottle.run(app, host="localhost", port="8080")
