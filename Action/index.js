
import { FETCH_USAGE_BY_ID_SUCCESS,FETCH_REFILL_LATEST,FETCH_USAGE_BY_COUNTER,FETCH_REFILL_VALUE } from './types';

import axios from 'axios';






const get_usage_by_counter = ()=>{
    
    return{
        type:FETCH_USAGE_BY_COUNTER,
        
    }

}


const fetchUsagesbyidSu =(usage) =>{
    return {
        type: FETCH_USAGE_BY_ID_SUCCESS,
        usage
    }
}
const fetchRefillValue =(refill) =>{
    return {
        type: FETCH_REFILL_VALUE,
        refill
    }
}  

const fetchRefillLatest =(latest) =>{
    return {
        type: FETCH_REFILL_LATEST,
        latest
    }
}  


export const get_usage_by_id = (usageid)=>{ 

    return function(dispatch){

        axios.get(`http://192.168.0.104:3001/api/v1/drive/${usageid}`).then((usage)=>{
           
            dispatch(fetchUsagesbyidSu(usage.data));
        }).catch((err)=>{console.log(err)});

    }
}
export const get_refill = ()=>{ 

    return function(dispatch){

        axios.get('http://192.168.0.104:3001/api/v1/refill/latest/').then((refill)=>{
            
            dispatch(fetchRefillValue(refill.data));
        }).catch((err)=>{console.log(err)});

    }
}

export const get_refill_latest = ()=>{ 

    return function(dispatch){

        axios.get('http://192.168.0.104:5000/refill/').then((latest)=>{
            
            dispatch(fetchRefillLatest(latest.data));
        }).catch((err)=>{console.log(err)});

    }
}
export const get_usage_by_row = ()=>{ 

        return function(dispatch){
               

                dispatch(get_usage_by_counter());
            }
    
        }    


