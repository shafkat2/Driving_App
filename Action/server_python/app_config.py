from flask import Flask,request,url_for,jsonify
from flask_pymongo import PyMongo
from Algorithm import*
from mongo_url import DB_URI


app = Flask(__name__)
app.config['MONGO_URI'] = DB_URI
mongo = PyMongo(app)
data = pd.DataFrame(list(mongo.db.drivings.find()))
data1 = Zero_noise(data,"Fuel_Information")
data1 = white_noise_filter(data1)
data1 = filter_iqr(data1,0,0.0005,200)
data1 = Clustering(data1,200,1,1)
data2 = Mini_batch_wavelet_denoising(data1,200,4,"coif1",5,12,5)
refill = detecting_refill_ref(data1,data2,90,100,lag= 1,threshold_neuro =1,refill_th_neuro=3,threshold=1,refill_th = 3)
data2  = filter_method(data1,refill,"median",10,9)
data1  = Mini_batch_wavelet_denoising(data2,100,2,"coif2",4,12,5)
refill = refill1 = detecting_refill_ref(data2,data1,90,100,lag= 1,threshold_neuro =2,refill_th_neuro=4,threshold=2,refill_th = 4)
data1 =  filter_method(data2,refill1,"median",10,9)
refill = remove_false_peaks(refill1,data1,diff = 2 ,lag = 1,threshold_stole = 1,refill_th = 3 ,ranges_ref = 10 ,ranges = 250 )
refill = Peak_analysis_IQR(data1,refill,0,1,0,1,"interp","both",1,0.100)
result = Usage_data(data1,refill,data['START_DATE'])
print(result)

# @app.route('/')
# def add_to_database():
#     Refills = mongo.db.refills
#     Refills.insert_many(result.to_dict('records'))
#     return 'added user'

@app.route('/refill/', methods=['GET'] )
def api_refill():
        Refills = mongo.db.refills
        Refills.insert_many(result.to_dict('records'))
        res = result.iloc[-2:-1].to_dict(orient='records')
        return jsonify(res)

if __name__ == '__main__':
    app.run(host='192.168.0.104', port=5000,debug = True)