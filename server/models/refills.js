const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const refillSchema = new Schema({
    
    Date_index:  Date,

    Refill_value: Number,
  
    After_refill_usage: Number,

    total_fuel_usage: Number


    
  }); 

module.exports = mongoose.model('Refill ',refillSchema,'refills')