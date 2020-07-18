import layerInfo from '../conf/layerInfo.js';
import dataset from '../conf/dataset.js';

export const state = {
  polygons: [],
  box: null
}

// needed to initiate the map
let groups = ["map", "controls"];
d3.select("#root")
  .selectAll("div")
  .data(groups).enter()
  .append("div")
  .attr("id", (d) => d);

let tiles = {
  "ESRI": L.tileLayer(
    layerInfo.ESRI.url,
    {attribution: ""}
  ),
  "prediction": L.tileLayer(
    dataset.predictionLayer.url,
    {tms: true}
  ),
  "2-4-5": L.tileLayer(
    dataset.basemapLayer.url,
    {tms: true}
  ),
};

export let map = L.map("map", {
  zoomControl: true,
  crs: L.CRS.EPSG3857, // this is the projection CRS (EPSG:3857), but it is different than the data CRS (EPSG:4326). See https://gis.stackexchange.com/questions/225765/leaflet-map-crs-is-3857-but-coordinates-4326/225786.
  center: dataset.basemapLayer.initialLocation,
  zoom: dataset.basemapLayer.initialZoom,
  minZoom: dataset.basemapLayer.args.minZoom,
  maxZoom: dataset.basemapLayer.args.maxZoom,
  fadeAnimation: false,
  layers: Object.values(tiles)
});

L.control.layers(tiles).addTo(map);
export let backendUrl = "http://localhost:4446/";

map.on("drag", function() {
  map.fire("viewreset");
});

map.on("zoomstart", function() {
  map.fire("viewreset");
});
