import * as d3f from './node_modules/d3-fetch/dist/d3-fetch.js';
import * as d3s from './node_modules/d3-selection/dist/d3-selection.min.js';
import * as d3sm from './node_modules/d3-selection-multi/build/d3-selection-multi.js';
import * as f from './funs.js';
import { map, backendUrl } from './globals.js';

// Setup the map
f.initializeMap();
f.addButtons("#controls");
map.on("zoom", f.redraw);

// d3f.json(backendUrl + "predTile", {
//   method: "POST",
//   headers: { "Content-Type": "application/json" },
//   body: JSON.stringify({
//     extent: {
//       xmin: coords._southWest.lng,
//       xmax: coords._northEast.lng,
//       ymin: coords._southWest.lat,
//       ymax: coords._northEast.lat,
//       crs: 3857
//     },
//     dataset: dataset,
//     classes: dataset["classes"],
//     models: models["benjamins_unet"]
//   })
// }).then((data) => displayPred(data));
