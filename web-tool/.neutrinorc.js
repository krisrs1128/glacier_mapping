const airbnbBase = require('@neutrinojs/airbnb-base');
const web = require('@neutrinojs/web');

module.exports = {
  options: {
    root: __dirname,
  },
  use: [
    airbnbBase(),
    web({
      html: {
        title: 'web-tool'
      }
    }),
  ],
};
