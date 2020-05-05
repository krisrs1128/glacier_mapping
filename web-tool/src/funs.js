import * as d3s from 'd3-selection';
import * as d3a from 'd3-array';
import * as d3sm from 'd3-selection-multi';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import { state, map } from './globals';
import * as d3g from 'd3-geo';
import layerInfo from '../conf/layerInfo';
import './map.css';

export function initializeMap() {
  // leaflet setup
  let tiles = L.tileLayer(
    layerInfo.ESRI.url,
    {attribution: ""}
  ).addTo(map);

  // add svg overlay
  L.svg({clickable:true}).addTo(map)
  const overlay = d3s.select(map.getPanes().overlayPane)
  overlay.select('svg')
    .attrs({
      "pointer-events": "auto",
      "id": "mapOverlay"
    });

  d3s.select("#mapOverlay")
    .selectAll("g")
    .data(d3a.range(10)).enter()
    .append("g")
    .attr("id", (d) => "polygon-" + (d - 1));
}

export function addButtons(parent_id) {
  d3s.select(parent_id)
    .append("button")
    .text("New Polygon")
    .on("click", newPoly);
}

function newPoly() {
  map.addEventListener("mousemove", nodeReposition);
  map.addEventListener("click", addNode);

  // update the polygon's state
  let poly = state.polygons;
  poly.push([]);
  state.polygons = poly;
  state.focus = poly.length - 1;
  state.mode = "create";
}

function addNode(event) {
  let mousePos = [event.latlng.lat, event.latlng.lng],
      poly = state.polygons;
  poly[state.focus].push(mousePos);
  state.polygons = poly;

  let curPoly = poly[state.focus];
  if (curPoly.length > 2 & dist2(curPoly[0], curPoly[curPoly.length - 1]) < 0.001) {
    curPoly.splice(-2, 2);
    poly[state.focus] = curPoly;
    map.removeEventListener("mousemove", nodeReposition);
    map.removeEventListener("click", addNode);
    state.polygons = poly;
    state.mode = "edit";
  }

}

function nodeReposition(event) {
  let mousePos = [event.latlng.lat, event.latlng.lng],
      poly = state.polygons,
      cur_poly = poly[state.focus];

  if (cur_poly.length == 0) {
    cur_poly.push(mousePos);
  } else if (cur_poly.length > 2 & dist2(mousePos, cur_poly[0]) < 0.001) {
    cur_poly[cur_poly.length - 1][0] = cur_poly[0][0];
    cur_poly[cur_poly.length - 1][1] = cur_poly[0][1];
  } else {
    cur_poly[cur_poly.length - 1][0] = mousePos[0];
    cur_poly[cur_poly.length - 1][1] = mousePos[1];
  }

  poly[state.focus] = cur_poly;
  state.polygons = poly;

  redraw();
}

function redraw() {
  let curPoly = state.polygons[state.focus];
  let pointPoly = curPoly.map((d) => map.latLngToLayerPoint(new L.LatLng(d[0], d[1])));

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll("circle")
    .data(pointPoly).enter()
    .append("circle")
    .attrs({
      class: "polyNode",
      cx: (d) => d[0],
      cy: (d) => d[1],
    });

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll(".polyNode")
    .data(pointPoly)
    .attrs({
      cx: (d) => d.x,
      cy: (d) => d.y
    });
}


function dist2(a, b) {
  return Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2);
}


function allScales(projector) {
  let transform = d3g.geoTransform({point: projector});
  let p = d3g.geoPath(transform);


  return {
    "p": p
  };
}

export function projectionFactory(map) {
  function projectPoint(x, y) {
    var point = map.latLngToLayerPoint(new L.LatLng(y, x));
    this.stream.point(point.x, point.y);
  }

  return projectPoint;
}
