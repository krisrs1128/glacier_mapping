import * as d3s from 'd3-selection';
import * as d3sm from 'd3-selection-multi';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import dataset from '../conf/dataset.json';
import { state } from './globals';
import './map.css';
import * as f from './funs';

/* Setup the map */
f.initializeMap("#root");
f.addButtons("#controls")

// on click on button, add new empty polygson.
// this polygon becomes the focus

// on click on map, the focus polygon gets a new node
//



// This is needed for Hot Module Replacement
if (module.hot) {
  module.hot.accept();
}
