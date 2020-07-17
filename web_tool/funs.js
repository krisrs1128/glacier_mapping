import { state, map, backendUrl } from './globals.js';
import dataset from '../conf/dataset.js';
import models from '../conf/models.js';



export function initializeMap() {
  map.pm.addControls({
    drawMarker: false,
    drawPolygon: true,
    editPolygon: true,
    drawPolyline: false,
    deleteLayer: true,
  });

  // add svg overlay
  L.svg({clickable:true}).addTo(map);
  const overlay = d3.select(map.getPanes().overlayPane);
  overlay.select('svg')
    .attrs({
      "pointer-events": "auto",
      "id": "mapOverlay"
    });

  map.on("keydown", function(event) {
    if (event.originalEvent.key == "Shift") {
      predictionExtent(event.latlng, "add");
    }
  });

  map.on("keyup", function(event) {
    if (event.originalEvent.key == "Shift") {
      predictionExtent(event.latlng, "add");
    }
  });
}

function removeListeners() {
  map.removeListeners();
}

function predictionExtent(latlng) {
  let box = L.polygon([[0, 0], [0, 0]], {"id": "predictionBox"});
  box.addTo(map);
  map.addEventListener("mousemove", extentMoved(box));
  map.addEventListener("keydown", removePatch(box));
  map.addEventListener("click", predPatch(box));
}

/*
 * Associate a Listener with an Extent
 *
 * We need a function factory because we need to associate our mousemove with a
 * function that has a single 'event' argument. However, that event needs to
 * refer to a previously instantiated extent / box. So, we return a function
 * that has access to the box in its scope.
 */
function extentMoved(box) {
  return function(event) {
    let box_coords = getPolyAround(event.latlng, 10000);
    box.setLatLngs(box_coords);
  };
}

function removePatch(box) {
  return function(event) {
    if (event.originalEvent.key == "Escape") {
      box.remove();
    }
  };
}

function predPatch(box) {
  return function(event) {
    const coords = box.getBounds();

    $.ajax({
      type: 'POST',
      url: "http://test.westus2.cloudapp.azure.com:8080/predPatch",
      contentType: "application/json",
      crossDomain:'true',
      dataType: "json",
      data: JSON.stringify({
        extent: {
          xmin: coords._southWest.lng,
          xmax: coords._northEast.lng,
          ymin: coords._southWest.lat,
          ymax: coords._northEast.lat,
          crs: 3857
        },
        classes: dataset["classes"],
        models: models["benjamins_unet"]
      }),
      success: function(response){
        displayPred(response);
      },
    });
  };
}

function decode_img(img_str) {
  return "data:image/jpeg;base64," + img_str;
}

function displayPred(data, show_pixel_map=false) {
  let coords = [[data.extent.ymin, data.extent.xmin],
                [data.extent.ymax, data.extent.xmax]];
  if (show_pixel_map) {
    L.imageOverlay(decode_img(data["output_soft"]), coords).addTo(map);
  }

  L.geoJSON(data["y_geo"], {
    pmIgnore: false
  }).addTo(map);
}

function getPolyAround(latlng, radius){
  // We convert the input lat/lon into the EPSG3857 projection, define our
  // square, then re-convert to lat/lon
  let latlngProjected = L.CRS.EPSG3857.project(latlng),
      x = latlngProjected.x,
      y = latlngProjected.y;

  let top = Math.round(y + radius/2),
      bottom = Math.round(y - radius/2),
      left = Math.round(x - radius/2),
      right = Math.round(x + radius/2);

  // left / right are "x" points while top/bottom are the "y" points
  let topleft = L.CRS.EPSG3857.unproject(L.point(left, top));
  let bottomright = L.CRS.EPSG3857.unproject(L.point(right, bottom));

  return [[topleft.lat, topleft.lng],
          [topleft.lat, bottomright.lng],
          [bottomright.lat, bottomright.lng],
          [bottomright.lat, topleft.lng]];
}


export function geomanControls() {
  map.pm.addControls({
    position: 'topleft',
    drawCircle: false,
  });
}
