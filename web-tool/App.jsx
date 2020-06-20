import { hot } from 'react-hot-loader';
import React, { Component } from 'react';
import './App.css';
import MapDisplay from './Map.jsx';
import Test from './Test.jsx';
import BasemapSelector from './BasemapSelector';
import 'leaflet/dist/leaflet.css';
import { Button } from 'antd';
import dataset from '../conf/dataset.json';

function dist2(a, b) {
  return Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2);
}

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      basemap: "ESRI",
      keyPressed: false,
      polygons: [],
      mousePos: dataset.basemapLayer.initialLocation,
    };
    this.addNode = this.addNode.bind(this);
    this.handleBasemapChange = this.handleBasemapChange.bind(this);
    this.handleDrawClick = this.handleDrawClick.bind(this);
    this.handleKeypress = this.handleKeypress.bind(this);
    this.handleKeyup = this.handleKeyup.bind(this);
    this.handleMove = this.handleMove.bind(this);
    this.newPoly = this.newPoly.bind(this);
  }

  componentDidMount(){
    document.addEventListener("keydown", this.handleKeypress, false);
    document.addEventListener("keyup", this.handleKeyup, false);
  }

  handleBasemapChange(event) {
    this.setState({basemap: event.target.value});
  }

  handleKeypress(event) {
    if (event.keyCode == 16) {
      this.setState({keyPressed: true});
    }
  }

  handleKeyup(event) {
    this.setState({keyPressed: false});
  }

  handleDrawClick(event) {
    this.setState({drawing: true});
  }

  handleMove(event) {
    this.setState({mousePos: event.latlng});
  }

  newPoly(event) {
    // document.addEventListener("mousemove", this.node_reposition);
    // document.addEventListener("click", this.add_node);
    console.log(this.state.mousePos)
    let poly = this.state.polygons;
    poly.push([]);
    this.setState({
      mode: "create",
      polygons: poly,
      focus: poly.length - 1
    });
  }

  addNode(event) {
    // let mouse_pos = coords(event);
    console.log(event.latlng)
    // let poly = this.state.polygons
    // poly[this.state.focus].push(mouse_pos);
    this.setState({polygons: poly});

    console.log(d3.select("#.Map"))

    let cur_poly = poly[this.state.focus];
    if (cur_poly.length > 2 & dist2(cur_poly[0], cur_poly[cur_poly.length - 1]) < 40) {
      cur_poly.splice(-2, 2);
      poly[this.state.focus] = cur_poly;
      // document.removeEventListener("mousemove", this.node_reposition);
      // document.removeEventListener("click", this.add_node);
      this.setState({polygons: poly, mode: "edit"})
    }
  }

  render() {
    return (
      <div>
        <div className="Map" onKeyPress={this.handleKeypress} onKeyUp={this.handleKeyup}>
          <MapDisplay basemap={this.state.basemap} keyPressed={this.state.keyPressed} drawing={this.state.drawing}/>
          <Test/>
        </div>
        <div className="Select">
          <BasemapSelector basemap={this.state.basemap} onChange={this.handleBasemapChange}/>
      </div>
        <div>
          <Button onClick={this.newPoly}>New Polygon</Button>
        </div>
      </div>
    );
  }
}

export default hot(module)(App);
