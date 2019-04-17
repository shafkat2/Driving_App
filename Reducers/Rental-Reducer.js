import { FETCH_USAGE_BY_ID_SUCCESS,FETCH_USAGE_BY_COUNTER,FETCH_REFILL_VALUE,FETCH_REFILL_LATEST  } from '../Action/types';





const Initial_state = {

    usage: {
        data :{}
    },
    refill:{
        data:[]
    },
    latest: {
        data :[]
    },
    row: 1
}


export const usageIDReducer = (state =  Initial_state.usage,action) => {
    switch(action.type){
        case FETCH_USAGE_BY_ID_SUCCESS:
            return {...state, data: action.usage};
        
           
        default:
            return state;
    }
}
export const refillReducer = (state =  Initial_state.refill,action) => {
    switch(action.type){
        case FETCH_REFILL_VALUE:
            
            return {...state, data: action.refill};
        
           
        default:
            return state;
    }
}

export const refillLatestReducer = (state =  Initial_state.latest,action) => {
    switch(action.type){
        case FETCH_REFILL_LATEST:
            console.log(state)
            return {...state, data: action.latest};
        
           
        default:
            return state;
    }
}


export const counter = (state = Initial_state,action) => {
    switch(action.type){
        case FETCH_USAGE_BY_COUNTER:
            
            return {...state, row: state.row + 1};
        
           
        default:
            return state;
    }
}