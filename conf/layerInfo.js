export default {
  "Custom": {
  },
  "OSM": {
    "args": {
      "minZoom": 8,
      "maxZoom": 14,
      "attribution": "&amp;copy <a href='http://osm.org/copyright'>OpenStreetMap</a> contributors"
    },
    "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
  },
  "OTM": {
    "args": {
      "minZoom": 8,
      "maxZoom": 14
    },
    "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
  },
  "ESRI": {
    "args": {
      "minZoom": 8,
      "maxZoom": 14
    },
    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
  }
}
