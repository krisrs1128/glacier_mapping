import layerInfo from '../../conf/layerInfo.js';
import dataset from '../../conf/dataset.js';

export const state = {
  polygons: [],
  box: null
}

export const tiles = {
  "ESRI_left": L.tileLayer(
    layerInfo.ESRI.url,
    {attribution: ""}
  ),
  "prediction_left": L.tileLayer(
    dataset.predictionLayer.url,
    {tms: true}
  ),
  "5-4-2_left": L.tileLayer(
    dataset.basemapLayer.url,
    {tms: true}
  ),
  "ESRI_right": L.tileLayer(
    layerInfo.ESRI.url,
    {attribution: ""}
  ),
  "prediction_right": L.tileLayer(
    dataset.predictionLayer.url,
    {tms: true}
  ),
  "5-4-2_right": L.tileLayer(
    dataset.basemapLayer.url,
    {tms: true}
  ),
};

export let map = L.map("map", {
  crs: L.CRS.EPSG3857, // this is the projection CRS (EPSG:3857), but it is different than the data CRS (EPSG:4326). See https://gis.stackexchange.com/questions/225765/leaflet-map-crs-is-3857-but-coordinates-4326/225786.
  center: dataset.basemapLayer.initialLocation,
  zoom: dataset.basemapLayer.initialZoom,
  minZoom: dataset.basemapLayer.args.minZoom,
  maxZoom: dataset.basemapLayer.args.maxZoom,
  fadeAnimation: false,
  layers: [tiles["5-4-2_left"], tiles["5-4-2_right"]]
});

export let backendUrl = "http://localhost:4446/";
