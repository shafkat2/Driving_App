const express = require('express');
const router = express.Router();
const drive = require('../models/drive');


router.get('/:id',function(req,res){
    const row_number = req.params.id;

    drive.findOne({ $or:[ {Row_Number:row_number}]},function(err,foundRental){
        if(err){
            res.status(422).send({error:[{title:'data error', detail:'could not find data'}]});
        }

    res.json(foundRental);
    });
});




module.exports = router;