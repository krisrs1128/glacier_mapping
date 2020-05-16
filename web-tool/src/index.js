import './map.css';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import * as d3f from 'd3-fetch';
import * as d3s from 'd3-selection';
import * as d3sm from 'd3-selection-multi';
import * as f from './funs';
import dataset from '../../conf/dataset.json';
import models from '../../conf/models.json';
import { state, map, backendUrl } from './globals';

// Setup the map
f.initializeMap("#root");
f.addButtons("#controls")
map.on("zoom", f.redraw)

// when the user presses shift, highlight a region on which to draw predictions

// when the user clicks, make a post to the predictions
d3s.select("#root")
  .append("button")
  .text("predict")
  .on("click", predFun);

function predFun() {
  d3f.json(backendUrl + "predPatch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      "extent": {"xmin": 80.5, "xmax": 80.8, "ymin": 26.7, "ymax": 27, "crs": 4326},
      "dataset": dataset,
      "classes": dataset["classes"],
      "models": models["benjamins_unet"]
    })
  }).then((data) => console.log(data));
}

// extract the geojson associated with a prediction


// This is needed for Hot Module Replacement
if (module.hot) {
  module.hot.accept();
}
