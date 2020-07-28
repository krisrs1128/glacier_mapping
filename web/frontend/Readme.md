
The main commands that come in handy are,

```
python3 frontend_server.py
python3 -m web_backend.server # from the glacier_mapping repo

lsof -i :4040 # to list the PID listening to this port
kill -9 <PID> # to open up that port again
lsof -i :4040 | awk '{l=$2} END {print l}' | xargs kill # get pid for the port and kill it
lsof -i :8080 | awk '{l=$2} END {print l}' | xargs kill # get pid for the port and kill it
```

Navigate to in browser

```
http://52.247.203.144:4040/web-tool/index.html
```
