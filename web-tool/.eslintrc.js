// const neutrino = require('neutrino');

// module.exports = neutrino().eslintrc();
const airbnbBase = require('@neutrinojs/airbnb-base');
const web = require('@neutrinojs/web');

module.exports = {
  options: {
    root: __dirname,
  },
  use: [false, web()],
};
