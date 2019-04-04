const express = require('express');
const router = express.Router();
const drive = require('../models/drive');


router.get('',function(req,res){
    drive.find({},function(err,foundDrive){

    res.json(foundDrive);
    });
});




module.exports = router;