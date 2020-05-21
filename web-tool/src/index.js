import './map.css';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import * as d3f from 'd3-fetch';
import * as d3s from 'd3-selection';
import * as d3sm from 'd3-selection-multi';
import * as f from './funs';
import map from './globals';

// Setup the map
f.initializeMap();
f.addButtons("#controls");
map.on("zoom", f.redraw);

// This is needed for Hot Module Replacement
if (module.hot) {
  module.hot.accept();
}
