const express = require("express"); 
const mongoose = require("mongoose");
const config = require("./config/def");
const fakedb = require("./Fake-db")
var Excel = require('exceljs');
const driverRoutes = require("./routes/drive");
const latestRoutes = require("./routes/latest");
var sleep = require('system-sleep');


const options = {
    useNewUrlParser: true,
}

mongoose.connect(config.DB_URI,options)
                    .then(()=>{
                            console.log("mongoDb Connected ....")
                            var workbook = new Excel.Workbook();
                            workbook.xlsx.readFile("./file_assets/capstone.xlsx").then(function(){
                            const worksheet = workbook.getWorksheet(1);                
                                          worksheet.eachRow((rows,rowNumber)=>{  
                                                if(rowNumber !== 1){
                                                console.log(rowNumber)    
                                                const fakeob = new fakedb(rows,rowNumber);
                                                sleep(1000); 
                                                fakeob.seedDB();
                                                }
                                                else{
                                                    console.log(rows.values)
                                                }
                                            }).catch(err =>console.log(err))
                                         
                                        
                                    }
                                            ).catch(err =>console.log(err))
                    })
                    .catch(err => console.log(err));

const app = express();


app.use('/api/v1/drive',driverRoutes);
app.use('/api/v1/refill',latestRoutes);



const PORT = process.env.PORT || 3001;



app.listen(PORT, function(){
    console.log("App is running....");
    
});


