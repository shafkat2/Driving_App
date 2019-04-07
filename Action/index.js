//import { fetch_rentals } from './types';
import { FETCH_USAGE_BY_ID_SUCCESS,FETCH_USAGE_BY_IDINIT,FETCH_USAGE_BY_COUNTER } from './types';

import axios from 'axios';



const get_usage_by_idinit = ()=>{
    
    return{
        type:FETCH_USAGE_BY_IDINIT
    }

}

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


export const get_usage_by_id = (usageid)=>{ 

    return function(dispatch){

        dispatch(get_usage_by_idinit());

        axios.get(`http://localhost:3000/api/v1/drive${usageid}`).then((usage)=>{
            
            dispatch(fetchUsagesbyidSu(usage.data));
        });

    }
}
export const get_usage_by_row = ()=>{ 

        return function(dispatch){
               

                dispatch(get_usage_by_counter());
            }
    
        }    


