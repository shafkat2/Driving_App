const Drive = require("./models/drive");


class fakeob{
    
   
   
    constructor(rows,rowNumber){
       
        this.drive =  [{ 
            Row_Number: rowNumber,
            START_DATE: rows.values[3],
            Status: rows.values[5],
            Velocity: rows.values[17],
            Battery_Status: rows.values[6],
            Voltage: rows.values[8],
            TravelDistance: rows.values[9],
            Coordinates: { Lat: rows.values[2],long:rows.values[1]},
            Fuel_Information: rows.values[22],
            }]
    }

    async cleandb(){
      await Drive.remove({});
    }

    pushDrivedb(){
            this.drive.forEach( (drive) =>{
            const newDrive = new Drive(drive);

            newDrive.save();
            } )
    }
    seedDB(){
        //this.cleandb();
        this.pushDrivedb();
    }


}
module.exports = fakeob;