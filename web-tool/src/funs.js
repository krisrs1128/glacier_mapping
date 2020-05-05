import * as d3s from 'd3-selection';
import * as d3a from 'd3-array';
import * as d3sm from 'd3-selection-multi';
import * as d3sh from 'd3-shape';
import 'leaflet';
import 'leaflet/dist/leaflet.css';
import { state, map } from './globals';
import * as d3g from 'd3-geo';
import layerInfo from '../conf/layerInfo';
import './map.css';

// some global operations
map.on("zoom", redraw)


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
    redraw();
  }
}

function nodeReposition(event) {
  let mousePos = [event.latlng.lat, event.latlng.lng],
      poly = state.polygons,
      curPoly = poly[state.focus];

  if (curPoly.length == 0) {
    curPoly.push(mousePos);
  } else if (curPoly.length > 2 & dist2(mousePos, curPoly[0]) < 0.001) {
    curPoly[curPoly.length - 1][0] = curPoly[0][0];
    curPoly[curPoly.length - 1][1] = curPoly[0][1];
  } else {
    curPoly[curPoly.length - 1][0] = mousePos[0];
    curPoly[curPoly.length - 1][1] = mousePos[1];
  }

  poly[state.focus] = curPoly;
  state.polygons = poly;

  redraw();
}

function nodeMove(event) {
  map.dragging.disable();
  let mousePos = [event.latlng.lat, event.latlng.lng],
      curPoly = state.polygons[state.focus];

  let ix = closestNode(curPoly, mousePos);
  curPoly[ix] = mousePos;
  let poly = state.polygons;
  poly[state.focus] = curPoly;
  state.polygons = poly;
  redraw();
}

function nodeDown(event) {
  if (state.mode != "create") {
    map.addEventListener("mousemove", nodeMove)
  }
}

function nodeUp(event) {
  if (state.mode != "create") {
    map.dragging.enable();
    map.removeEventListener("mousemove", this.nodeMove);
  }
}

function redraw() {
  let curPoly = state.polygons[state.focus];
  let pointPoly = curPoly.map((d) => map.latLngToLayerPoint(new L.LatLng(d[0], d[1])));
  pointPoly = pointPoly.map((d) => [d.x, d.y]);

  // drawing the polygon nodes
  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll("circle")
    .data(pointPoly).enter()
    .append("circle")
    .attrs({
      class: "polyNode",
      cx: (d) => d[0],
      cy: (d) => d[1],
    })
    .on("mouseup", nodeUp)
    .on("mousedown", nodeDown);

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll(".polyNode")
    .data(pointPoly)
    .attrs({
      cx: (d) => d[0],
      cy: (d) => d[1]
    });

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll(".polyNode")
    .data(pointPoly).exit()
    .remove();

  // draw the polygon edges
  let line = d3sh.line()
      .x((d) => d[0])
      .y((d) => d[1]);

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll(".polyEdge")
    .data([pointPoly]).enter()
    .append("path")
    .attrs({
      "d": line,
      "class": "polyEdge"
    });

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll(".polyEdge")
    .data([pointPoly])
    .attrs({
      "d": line
    });

  d3s.select("#mapOverlay")
    .select("#polygon-" + state.focus)
    .selectAll(".polyEdge")
    .data([pointPoly]).exit()
    .remove();
}


function dist2(a, b) {
  return Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2);
}

function closestNode(poly, pos) {
  let ix = 0,
      min_dist = Infinity;

  for (var i = 0; i < poly.length; i++) {
    let dist = dist2(poly[i], pos);
    if (dist < min_dist) {
      min_dist = dist;
      ix = i;
    }
  }
  return ix;
}
