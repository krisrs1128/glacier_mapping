import './styles/map.css';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import * as d3f from 'd3-fetch';
import * as d3s from 'd3-selection';
import * as d3sm from 'd3-selection-multi';
import * as f from './funs';
import { map } from './globals';

// Setup the map
f.initializeMap();
f.addButtons("#controls");
map.on("zoom", f.redraw);

d3f.json(backendUrl + "predTile", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    extent: {
      xmin: coords._southWest.lng,
      xmax: coords._northEast.lng,
      ymin: coords._southWest.lat,
      ymax: coords._northEast.lat,
      crs: 3857
    },
    dataset: dataset,
    classes: dataset["classes"],
    models: models["benjamins_unet"]
  })
}).then((data) => displayPred(data));
