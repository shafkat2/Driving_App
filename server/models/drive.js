const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const DrivingSchema = new Schema({
    Row_Number: {

      type: Number

    },
    START_DATE: {

      type: Date
  
    },
    Status: {

      type: Boolean,
      required: true
    },
    Velocity: {

      type: Number,
      required: true
    },
    Battery_Status: {

      type: Boolean,
      required: true
    },
    Voltage: {
 
      type: Number,
      required: true
    },
    TravelDistance: {

      type: Number,
      required: true
    },
    Coordinates: {

      type: {
        Lat: {

          type: Number,
        },
        long: {

          type: Number,
        }
      }
    },
    Fuel_Information: {

      type: Number,
      required: true
    },
  
    
  }); 

module.exports = mongoose.model('Driving',DrivingSchema)