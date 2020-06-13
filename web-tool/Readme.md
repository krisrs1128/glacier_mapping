
The main commands that come in handy are,

```
node webServer.js & npm run build:w # from web-tool directory
lsof -i :4040 # to list the PID listening to this port
kill -9 <PID> # to open up that port again
python3 -m web_backend.server # from the glacier_mapping repo
curl --data "param1=value1&param2=value2" http://localhost:4446/predTile # to test the backend endpoint
```
