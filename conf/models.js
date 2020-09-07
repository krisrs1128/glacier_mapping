export default {
  "shimas_unet": {
    "metadata": {
      "displayName": "UNet Shima Kind Durian 140"
    },
    "model": {
      "type": "pytorch",
      "numParameters": 7790949,
      "inputShape": [
        512,
        512,
        10
      ],
      "fn": "../data/models/model_188.pt",
      "fineTuneLayer": 0
    }
  },
  "benjamins_unet": {
    "metadata": {
      "displayName": "Benjamins model"
    },
    "model": {
      "type": "pytorch",
      "numParameters": null,
      "args": {
        "inchannels": 3,
        "outchannels": 1,
        "net_depth": 5,
        "channel_layer": 16
      },
      "inputShape": [
        512,
        512,
        3
      ],
      "fn": "runs/minimal_run/models/model_final.pt",
      "fineTuneLayer": 0,
      "process": "conf/postprocess.yaml"
    }
  }
}
