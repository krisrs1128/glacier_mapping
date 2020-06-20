import * as f from './funs.js';
import { map, backendUrl } from './globals.js';

// Setup the map
f.initializeMap();
f.addButtons("#controls");
map.on("zoom", f.redraw);

