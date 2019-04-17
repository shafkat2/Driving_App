import os
import numpy as np
import pandas as pd
import random
from scipy.interpolate import interp1d
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from scipy.signal import medfilt
from scipy.signal import savgol_filter
import pywt
from statsmodels.robust import mad
from sklearn.cluster import SpectralClustering
import uuid

from scipy.signal import sosfiltfilt, butter






count = 1  #global variable

#helper_function
def Zero_noise(df,X):
  
  da = df[X] == 0
 
  dm = df.copy(deep=True)
  dm[da] = None
  df = dm[X].iloc[:]
  df = pd.DataFrame(df)
  ward_list = fill_nans_scipy1(df.values)
  ward_list = pd.DataFrame(ward_list)

  return ward_list

def refill_concatenate(df,x,col,ranges):
  global count
  group = "Refill "
  if x+1 <= df.shape[0]-1:
    if df[col].iloc[x+1] - df[col].iloc[x] <= ranges:
      
      return count     # count is used a counting mechanism for counting the index
    else:
      count = count +1
      return count-1
  else:
      return count



def fill_nans_scipy1(padata, pkind='nearest'):
  #interpolates and expterpolates to calculate the values
  if(np.isnan(padata).any()):
    aindexes = np.arange(padata.shape[0])          
    agood_indexes = np.where(np.isfinite(padata))
    b = padata[agood_indexes[0]].squeeze()
    a = agood_indexes[0]
    f = interp1d(a, b, bounds_error=False, copy=False, fill_value="extrapolate",kind=pkind)
    
    return f(aindexes)
  else:
    return padata


def filter(a,window_median,polyorder,window_golay,mode,filters):
  #median and sav_golay_filter
  #a.iloc[1,0] 0 column and 1 index
  X_med = []
  if(filters == "median"):
    
    for i in range(0, a.shape[0]):
        X_med.append(a.iloc[i,0])
    X_med = medfilt(X_med,window_median)
    X_med = pd.DataFrame(X_med)
    
  elif(filters == "sav"):
    
    for i in range(0, a.shape[0]):
        X_med.append(a.iloc[i,0])
    X_med  = savgol_filter(X_med ,window_length=window_golay,polyorder= polyorder,mode = mode )
    X_med = pd.DataFrame(X_med)
    
  elif(filters == "both"):
    
    for i in range(0, a.shape[0]):
       X_med.append(a.iloc[i,0])
        
    X_med = medfilt(X_med,window_median)


    X_med  = savgol_filter(X_med ,window_length=window_golay,polyorder= polyorder,mode = mode )
    X_med = pd.DataFrame(X_med)
    X_med =  X_med.astype('float64')
  
  return X_med

def filter_method(S,refill_aggregate,filters,wl,div):  
  #start is bigger stop is smaller,
  # indexing dont count the last index
  S_test_final = pd.DataFrame()
  X_start = 0
  ranges = 0
  if refill_aggregate.shape[0] > 0:
    for x in range(refill_aggregate.shape[0]):
      if int((refill_aggregate["Index_start"].iloc[x]-X_start-wl)/div)% 2 == 0:
        ranges = int((refill_aggregate["Index_start"].iloc[x]-X_start-wl)/div)-1
      else:
        ranges = int((refill_aggregate["Index_start"].iloc[x]-X_start-wl)/div)
 
      S_test1 = filter(S[ X_start:refill_aggregate["Index_start"].iloc[x]-wl],window_median = ranges ,polyorder = 1,window_golay = ranges,mode = "interp",filters = filters)
      S_test2 = S[refill_aggregate["Index_start"].iloc[x]-wl:refill_aggregate["Index_stop"].iloc[x]+wl]
      X_start = refill_aggregate["Index_stop"].iloc[x]+wl
      S_list = [S_test1,S_test2]
      for x in range(len(S_list)):
        S_test_final = S_test_final.append(S_list[x], ignore_index=True)
    if int(( S.shape[0] - (refill_aggregate["Index_stop"].iloc[refill_aggregate.shape[0]-1]+wl))/div)% 2 == 0:
      ranges = int(( S.shape[0] - (refill_aggregate["Index_stop"].iloc[refill_aggregate.shape[0]-1]+wl))/div)-1
    else:
      ranges = int(( S.shape[0] - (refill_aggregate["Index_stop"].iloc[refill_aggregate.shape[0]-1]+wl))/div)
    #shape refill is 0 no peak detected  and -1 for not getting out of bounds to select the last point
    S_test1 = filter(S[refill_aggregate["Index_stop"].iloc[refill_aggregate.shape[0]-1]+wl:],window_median = ranges ,polyorder = 1,window_golay = ranges,mode = "interp",filters = filters)    
    S_test_final = S_test_final.append(S_test1, ignore_index=True)
  else:
    if int(( S.shape[0] - (X_start))/div)% 2 == 0:
      ranges = int(( S.shape[0] - (X_start))/div)-1
    else:
      ranges = int(( S.shape[0] - (X_start))/div)
      
      
    S_test1 = filter(S[X_start:],window_median = ranges ,polyorder = 1,window_golay = ranges,mode = "interp",filters = filters)
    S_test_final = S_test_final.append(S_test1, ignore_index=True)
  return  S_test_final

def new_feature_analysis(Volume_new_Feature):
  #add a new feature volume
  Volume = Volume_new_Feature.copy(deep = True)
  volume_np = [0]
  for i in range(Volume.shape[0]-1):
        volume_np.append(Volume.iloc[i+1, 0]-Volume.iloc[i, 0])

  vol = pd.DataFrame(volume_np)
  Volume = Volume.assign(volume  = vol)

  return Volume

###CLustering Part1

def Clustering(Data,batch_size,alpha,tuneing): 
  #Hiearchal CLustering
  batch_s = batch_size
  ward_list = []
  label = []
  mini_batch = int(batch_size/tuneing) 
  size1 = 0
  size2 =  batch_s
  data = []        
  count_row_test = Data.shape[0]
  num_batches =int(round(count_row_test/ batch_s)) 
  for x in range(num_batches):
      
    batch2 = Data.iloc[size1:size2]
    size_new1 = size1
    size_new2 = size1+mini_batch
    if(batch2.std().values<alpha):
      for i in range(tuneing):
        batch2 = Data.iloc[size_new1:size_new2]
        batch_noise = batch2.values.reshape( batch2.shape[0],1)
        if(batch2.shape[0]> 1):
          #connectivity = kneighbors_graph(batch_noise, n_neighbors=batch2.shape[0]-1, include_self=False)
          ward = SpectralClustering(n_clusters=2,  assign_labels="discretize", random_state=0).fit(batch_noise)
          ward_list.append(ward)
          data.append(batch2.values)
          label.append(ward.labels_)         
          size_new1 = size_new1+mini_batch
          size_new2 = size_new2+mini_batch
    else:
      batch_noise = batch2.values.reshape( batch2.shape[0],1)
      connectivity = kneighbors_graph(batch_noise, n_neighbors=batch2.shape[0]-1, include_self=False)
      ward = AgglomerativeClustering(n_clusters=2,connectivity=connectivity, linkage='single').fit(batch_noise)
      ward_list.append(ward)
      data.append(batch2.values)
      label.append(ward.labels_)

   
    size1 = size1+ batch_s
    size2 = size2+ batch_s

  #creating dataset
  label1 = np.concatenate(np.array( label))
  data1 =np.concatenate(np.array( data))
  db = pd.DataFrame({ 'labels':label1})
  db = db.assign(data = data1)
  da = db['labels'] == 0
  dc = db['labels'] == 1


  da = db[da] #up
  dc = db[dc]

  #sns.distplot(da['data'],bins=30)
  

  
  #removing noise
  da = db['labels'] == 0
  dc = db['labels'] == 1
 
  dm = db.copy(deep=True)
  dm[dc] = None
  d_test_noise = dm.iloc[:,1]
  d_test_noise = pd.DataFrame(d_test_noise)
  

  
  #Getting Indexes for interpolation
 
  d_test_noise = d_test_noise.values

  #interpolation
  New_test_data = fill_nans_scipy1(d_test_noise)
  New_test_data = pd.DataFrame(New_test_data)


  return New_test_data




#haar wavelet part3

def denoise(data,wavelet,levels,level,alpha):
    WC = pywt.wavedec(data,wavelet,mode = "per",level = level)
    noiseSigma = mad( WC[-levels] )
    threshold=alpha*noiseSigma*math.sqrt(math.log2(data.size))
    NWC = map(lambda x: pywt.threshold(x,threshold,'garotte'), WC)
    NWC = list(NWC)
    NWC = pywt.waverec( NWC, wavelet,mode= "per")
    NWC = pd.DataFrame(NWC)
    return NWC
  
  
def wavlet_analysis(Data,batch_size,wavelet,levels,level,alpha): 
  #wavelet analysis
  batch_s = batch_size
  ward_list = pd.DataFrame()
  size1 = 0
  size2 =  batch_s     
  count_row_test = Data.shape[0]
  num_batches =int(round(count_row_test/ batch_s)) 
  for x in range(num_batches):
    
    batch2 = Data.iloc[size1:size2]
    batch_noise = batch2.values.squeeze()
    ward = denoise(batch_noise,wavelet,levels,level,alpha)
    ward_list = ward_list.append(ward,ignore_index=True)
    size1 = size1+ batch_s
    size2 = size2+ batch_s
  ward_list = ward_list.values
  #interpolation
  ward_list = fill_nans_scipy1(ward_list)
  ward_list = pd.DataFrame(ward_list)

  return ward_list

def wavelet_analysis_method(S,refill_aggregate,wavelet,wl,level,levels,alpha):  
  S_test_final = pd.DataFrame()
  X_start = 0
  ranges = 0
  if(refill_aggregate.shape[0])>0:
    for x in range(refill_aggregate.shape[0]):
      ranges = refill_aggregate["Index_start"].iloc[x]- X_start
      S_test1 = wavlet_analysis(S[ X_start:refill_aggregate["Index_start"].iloc[x]-wl],ranges,wavelet,level,levels,alpha)
      S_test2 = S[refill_aggregate["Index_start"].iloc[x]-wl:refill_aggregate["Index_stop"].iloc[x]+wl]
      X_start = refill_aggregate["Index_stop"].iloc[x]+wl
      S_list = [S_test1,S_test2]
      for x in range(len(S_list)):
        S_test_final = S_test_final.append(S_list[x], ignore_index=True)
    ranges = S.shape[0] - (refill_aggregate["Index_stop"].iloc[refill_aggregate.shape[0]-1]+wl)
    S_test1 = wavlet_analysis(S[refill_aggregate["Index_stop"].iloc[refill_aggregate.shape[0]-1]+wl:],ranges,wavelet,level,levels,alpha)    
    S_test_final = S_test_final.append(S_test1, ignore_index=True)
  else:
    ranges = S.shape[0] - X_start
    S_test1 = wavlet_analysis(S[X_start:],ranges,wavelet,level,levels,alpha)    
    S_test_final = S_test_final.append(S_test1, ignore_index=True)
 
  return  S_test_final


def Mini_batch_wavelet_denoising(data,batch,batch_tweak,wavelet,level_to_denoise,level_of_localization,alpha):
  x1 = 0
  x2 = batch
  denoised = pd.DataFrame()
  for x in range(int(data.shape[0]/batch)):

    T  = wavlet_analysis(data[0].iloc[x1:x2],int(data[x1:x2].shape[0]/batch_tweak),wavelet,level_to_denoise,level_of_localization,alpha)
    denoised = denoised.append(T,ignore_index= True)
    x1 = x1 + batch
    x2 = x2 + batch



  return denoised
  
#Peak detection for stolen and Refill
def thresholding_algo_RF(y, lag, threshold_for_refill,ranges, influence):
    y = np.array(y)
    signals = np.zeros(len(y))
    Signal_RF = []
    Index_start_RF = []
    Index_stop_RF = []
    volume_RF = []
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    global count
    
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold_for_refill:
            if y[i] < avgFilter[i-1]:
                signals[i] = -1
                Signal_RF.append("refill")
                Index_start_RF.append(i-1)
                Index_stop_RF.append(i)
                volume_RF.append(y[i])  

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
      
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
    refills = dict(signals = Signal_RF, Index_start = Index_start_RF,Index_stop = Index_stop_RF,volume = volume_RF )
    refills = pd.DataFrame.from_dict(refills)
    # counts where the refill starts where end also includes noise
    refill_aggregate = refills.groupby(lambda x: refill_concatenate(refills,x,"Index_start",ranges)).aggregate({'Index_start': 'min','Index_stop': 'max','volume':'sum'})
    count = 1
    
    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter)),refills,refill_aggregate
  
  
def Refill(ranges,data,lag,threshold1,influence):
  result,Refill,Refill_aggregate = thresholding_algo_RF(data, lag=lag, threshold_for_refill=threshold1,ranges=ranges,influence=influence,)

  return Refill,Refill_aggregate
  
def detecting_Refill_raw(New_feature,lag,threshold1,refill_th,ranges):
  refill,refill_aggregate = Refill(ranges,New_feature["volume"],lag =lag,threshold1 = threshold1,influence = 0)
  refill_aggregate =refill_aggregate[refill_aggregate['volume'] < - refill_th ].reset_index()
  refill_aggregate.drop("index",axis = 1)
  return refill_aggregate

def thresholding_algo_ST(y,lag,threshold_for_steal, influence,ranges):
    signals = np.zeros(len(y))
    Signal_ST = []
    Index_start_ST = []
    Index_stop_ST = []
    volume_ST = []
    global count
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold_for_steal:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
                Signal_ST.append("stolen")
                Index_start_ST.append(i-1)
                Index_stop_ST.append(i)
                volume_ST.append(y[i])  

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
      
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
    stolen = dict(signals = Signal_ST, Index_start = Index_start_ST,Index_stop = Index_stop_ST,volume = volume_ST )
    stolen = pd.DataFrame.from_dict(stolen)
    stolen_aggregate = stolen.groupby(lambda x: refill_concatenate(stolen,x,"Index_start",ranges)).aggregate({'Index_start': 'min','Index_stop': 'max','volume':'sum'}) # groupby groups values passed on idex
    count = 1    

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter)),stolen,stolen_aggregate

def Stolen(ranges,data,lag =5,threshold1 = 2,influence = 0):
  result,Stolen,stolen_aggregate = thresholding_algo_ST(data, lag,threshold1, influence,ranges)

  return Stolen,stolen_aggregate


def detecting_Stolen(New_feature,lag,threshold1,stolen_th,ranges):
  stolen,stolen_aggregate = Stolen(ranges,New_feature["volume"],lag =lag,threshold1 = threshold1,influence = 0)
  stolen_aggregate =stolen_aggregate[stolen_aggregate['volume'] >  stolen_th ].reset_index()
  stolen_aggregate.drop("index",axis = 1)
  return stolen_aggregate  

 

def detecting_refill_ref(S,neural,ranges,ranges_ref,lag,threshold_neuro,refill_th_neuro,threshold,refill_th):
  
  New_Feature =  new_feature_analysis(S)
  vol =  new_feature_analysis(neural)
  
  refill_aggregate_raw = detecting_Refill_raw(New_Feature,lag,threshold,refill_th,ranges_ref)
  refill_aggregate_neuro = detecting_Refill_raw(vol,lag,threshold_neuro,refill_th_neuro,ranges_ref)
  

  Index_start = []
  Index_stop = []
  volume = []
  for x in range(len(refill_aggregate_neuro)):
    for y in range(len(refill_aggregate_raw)):
      if abs(refill_aggregate_neuro["Index_start"].iloc[x]-refill_aggregate_raw["Index_start"].iloc[y])<ranges:
        
        Index_start.append(refill_aggregate_raw["Index_start"].iloc[y])
        Index_stop.append(refill_aggregate_raw["Index_stop"].iloc[y])
        volume.append(refill_aggregate_raw["volume"].iloc[y])
      
          
  refill = dict( Index_start = Index_start,Index_stop = Index_stop,volume = volume )
  refill = pd.DataFrame.from_dict(refill)
  refill = refill.drop_duplicates(subset=None, keep='first', inplace=False)
    
  return refill






def Peak_analysis_IQR(Si,refill,column,window_median,polyorder,window_golay,mode,filters,order,fre):
  
  sos = butter(order, fre, output='sos')
  R = pd.DataFrame()
  Index_start = []
  Index_stop = []
  refills     = []
  for x in range(refill.shape[0]):
    if abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) > 1 and abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) <= 60 :
      R = Si[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+45]
      max_ = R.idxmax().values
      Index_start.append(max_[0])
      min_ = R.idxmin().values
      Index_stop.append(min_[0])
      refill_ = R.min().values - R.max().values
      refills.append(refill_[0])     
      
    elif abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) > 60 and abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) <= 100 :     
      R = Si[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+105]
      index = Si[0].iloc[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+105].index.tolist()
      index = np.array(index)
      Clean = filter(R,window_median = window_median ,polyorder = polyorder,window_golay = window_golay,mode = "interp",filters = filters)
      Clean = Clean.set_index(index)
      max_ = Clean.idxmax().values
      Index_start.append(max_[0])
      min_ = Clean.idxmin().values
      Index_stop.append(min_[0])
      volume = Clean.min().values - Clean.max().values
      refills.append(volume[0])

      
    elif abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) > 100 and abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) <= 180 :
      R = Si[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+165]
      index = Si[0].iloc[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+165].index.tolist()
      index = np.array(index)
      Clean = sosfiltfilt(sos, R.values.flatten())
      Clean = Clean.tolist()
      Clean = pd.Series(Clean, name=0, index=index)
      max_ = Clean.idxmax()
      Index_start.append(max_)
      min_ = Clean.idxmin()
      Index_stop.append(min_)
      volume = Clean.min() - Clean.max()
      refills.append(volume)
      
    elif abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) > 180 and abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) <= 240 :
      R = Si[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+225]
      index = Si[0].iloc[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+225].index.tolist()
      index = np.array(index)
      Clean = sosfiltfilt(sos, R.values.flatten())
      Clean = Clean.tolist()
      Clean = pd.Series(Clean, name=0, index=index)
      max_ = Clean.idxmax()
      Index_start.append(max_)
      min_ = Clean.idxmin()
      Index_stop.append(min_)
      volume = Clean.min() - Clean.max()
      refills.append(volume)
      
    elif abs(refill["Index_start"].iloc[x] - refill["Index_stop"].iloc[x]) > 240:
      R = Si[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+285]
      index = Si[0].iloc[refill["Index_start"].iloc[x]-15:refill["Index_start"].iloc[x]+285].index.tolist()
      index = np.array(index)
      Clean = sosfiltfilt(sos, R.values.flatten())
      Clean = Clean.tolist()
      Clean = pd.Series(Clean, name=0, index=index)
      max_ = Clean.idxmax()
      Index_start.append(max_)
      min_ = Clean.idxmin()
      Index_stop.append(min_)
      volume1 = Clean.min() - Clean.max()
      refills.append(volume1)

    else:
      Index_start.append(refill["Index_start"].iloc[x])
      Index_stop.append(refill["Index_stop"].iloc[x])
      refills.append(refill["volume"].iloc[x])
      
      
  detected = dict( Index_start = Index_start,Index_stop = Index_stop,volume= refills )
  detected = pd.DataFrame.from_dict(detected)

  return detected




  
#miscellanous  
def Usage_data(S,refill_aggregate,dp):  
  #start is bigger stop is smaller,
  # indexing dont count the last index

  S_new = S.copy(deep = True)

  S_test1 = 0
  S_total = 0
  X_start = 0

  refill_value = []
  After_refill_usage = []
  total_fuel_usage = []
  Date = []
  
  
  if refill_aggregate.shape[0] > 0:
    for x in range(refill_aggregate.shape[0]):
      S_test1 =S_new[0].loc[refill_aggregate["Index_start"].loc[x]] - S_new[0].loc[X_start]
      
      X_start = refill_aggregate["Index_stop"].loc[x]
      S_total = S_total+ S_test1
      
     
      refill_value.append(refill_aggregate["volume"].loc[x])
      Date.append(dp.loc[refill_aggregate["Index_start"].loc[x]])
      After_refill_usage.append(S_test1)
      total_fuel_usage.append(S_total)
      S_test1 = 0
    #shape refill is 0 no peak detected  and -1 for not getting out of bounds to select the last point
    S_test1 =S_new[0].loc[S_new.shape[0]-1] - S_new[0].loc[refill_aggregate["Index_stop"].loc[refill_aggregate.shape[0]-1]]
    S_total = S_total+ S_test1
      
    refill_value.append(0)
    Date.append(0)
    After_refill_usage.append(S_test1)
    total_fuel_usage.append(S_total)
  else:
    X_start = 0
    S_test1 =S_new[0].loc[X_start] - S_new[0].loc[S_new.shape[0]-1]
    S_total = S_total + S_test1
      
    refill_value.append(0)
    Date.append(0)
    After_refill_usage.append(S_test1)
    total_fuel_usage.append(S_total)
  usage_data = dict(Date_index = Date, Refill_value = refill_value,After_refill_usage = After_refill_usage,total_fuel_usage = total_fuel_usage )  
  
  usage_data = pd.DataFrame.from_dict(usage_data)
  return  usage_data


def filter_iqr(df,column,r1,batch):
    outlier_remove= pd.DataFrame()
    min_value  = 0
    max_value  = batch
    dk = df.copy(deep=True)
    for x in range(int(df.shape[0]/batch)):
      ds = dk[column].iloc[min_value:max_value].tolist()
      ds = pd.DataFrame(ds)
      q1 = ds[column].quantile(r1)
      iqr = (ds[column] <= q1) 

      ds[iqr] = None
      d_test_noise = ds.loc[:]
      outlier_remove = outlier_remove.append(d_test_noise)
      min_value = min_value+ batch
      max_value = max_value+ batch
    d_test_noise = pd.DataFrame(outlier_remove)



    #Getting Indexes for interpolation
    d_test_noise = d_test_noise.values



    #interpolation
    New_test_data = fill_nans_scipy1(d_test_noise)
    New_test_data = pd.DataFrame(New_test_data)

    return New_test_data
  
def remove_false_peaks(refill,data,diff,lag,threshold_stole,refill_th,ranges_ref,ranges):
  upward =  new_feature_analysis(data)


  upward_noise = detecting_Stolen(upward,lag,threshold_stole,refill_th,ranges_ref)
  

  Index_start = []
  Index_stop = []
  volume = []
  for x in range(len(refill)):
    for y in range(len(upward_noise)):
      if (abs(refill["Index_stop"].iloc[x]-upward_noise["Index_start"].iloc[y])< ranges) and abs((abs(refill["volume"].iloc[x])-upward_noise["volume"].iloc[y])<diff):
        Index_start.append(refill["Index_start"].iloc[x])
        Index_stop.append(refill["Index_stop"].iloc[x])
        volume.append(refill["volume"].iloc[x])
      else:  
        pass
      
          
  refill_false = dict( Index_start = Index_start,Index_stop = Index_stop,volume = volume )
  refill_false = pd.DataFrame.from_dict(refill_false)
  refill_false = refill_false.drop_duplicates(subset=None, keep='first', inplace=False)

  
  
  for x in range(refill_false.shape[0]):
   conditions = np.logical_and( refill["Index_start"] ==  refill_false["Index_start"].iloc[x],refill["Index_stop"] ==  refill_false["Index_stop"].iloc[x])
   refill = refill[~conditions]
  
  refill = refill.reset_index().drop(["index"], axis=1)
  

  return refill

def white_noise_filter(data):
  data1 = data.copy(deep=True)
  white_noise =  new_feature_analysis(data)
  

  var1 =  0

  var2 =  0
  
  for x in range(data.shape[0]-1):
      if (white_noise["volume"].iloc[x]>0) or (white_noise["volume"].iloc[x]< 0):
        var1 = white_noise["volume"].iloc[x]
        index11 = x
        var2 = white_noise["volume"].iloc[x+1]

        if ((var1+var2)/2) == 0:
          data1.iloc[index11] = None



  data1 = data1.interpolate()
  
  return data1    

  
    