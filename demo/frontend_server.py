#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''Easy HTTP server for serving the front end
'''
import sys
import os
import bottle
import argparse


@bottle.get("/")
def root_app():
    return bottle.static_file("index.html", root="./")


def main():
    parser = argparse.ArgumentParser(description="Frontend Server")
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="localhost")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default="4040")
    args = parser.parse_args(sys.argv[1:])

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "server": "tornado",
        "reloader": False
    }
    bottle.run(**bottle_server_kwargs)
    return


if __name__ == '__main__':
    main()
