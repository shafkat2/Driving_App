import { FETCH_USAGE_BY_ID_SUCCESS,FETCH_USAGE_BY_IDINIT,FETCH_USAGE_SUCCESS,FETCH_USAGE_BY_COUNTER  } from '../Action/types';
const Initial_state = {

    usage: {
        data :{}
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


export const counter = (state =  Initial_state,action) => {
    switch(action.type){
        case FETCH_USAGE_BY_COUNTER:
            return {...state, row: state.row + 1};
        
           
        default:
            return state;
    }
}