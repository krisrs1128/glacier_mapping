import * as f from './funs.js';
import { map, backendUrl } from './globals.js';

// Setup the map
f.initializeMap();
map.on("zoom", f.redraw);
