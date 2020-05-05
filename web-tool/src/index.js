import './map.css';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import * as d3f from 'd3-fetch';
import * as d3s from 'd3-selection';
import * as d3sm from 'd3-selection-multi';
import * as f from './funs';
import dataset from '../../conf/dataset.json';
import { state, map } from './globals';

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
  d3f.json("http://localhost:4446/predPatch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      "extent": [[0, 1], [2, 3]],
      "dataset": dataset,
      "classes": dataset["classes"]
    })
  }).then((data) => console.log(data));
}

// extract the geojson associated with a prediction


// This is needed for Hot Module Replacement
if (module.hot) {
  module.hot.accept();
}
