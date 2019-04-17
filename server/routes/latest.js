const express = require('express');
const router = express.Router();
const Refill = require('../models/refills');





router.get('/latest/',function(req,res){
  

    Refill.find({}).skip(Refill.count()-4).sort({ $natural: -1 }).limit(4).exec(function(err,foundRental){
        if(err){
            res.status(422).send({error:[{title:'data error', detail:'could not find data'}]});
        }
    
        res.json(foundRental);
    });
    
   
});


module.exports = router;