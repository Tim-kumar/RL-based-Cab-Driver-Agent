# Import routines

import numpy as np
import math
import random
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
loc = tuple([i+1 for i in range(m)])


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [elm for elm in  product(loc,loc)  if elm[0]!=elm[1]]
        self.state_space = [list(elm) for elm in  product(loc, np.arange(0,t), np.arange(0,d))]
        self.state_init =  random.choice(self.state_space)
        self.state_size = m+t+d


    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN."""
        state_encod=np.zeros(self.state_size)
        state_encod[state[0]-1]=1
        state_encod[m+state[1]]=1
        state_encod[m+t+state[2]]=1
        
        return state_encod



    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1 : requests = np.random.poisson(2)
        elif location == 2 : requests = np.random.poisson(12)
        elif location == 3 : requests = np.random.poisson(4)
        elif location == 4 : requests = np.random.poisson(7)
        elif location == 5 : requests = np.random.poisson(8)


        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(0, (m-1)*m ), requests) # (0,0) is not considered as customer request
        
        actions = [self.action_space[i] for i in possible_actions_index]
        
        possible_actions_index.append(len(self.action_space))
        actions.append((0,0))

        return possible_actions_index,actions   
    
    @staticmethod
    def transition(start,end,time,day,Time_matrix):
        next_state_loc=end
        if start==end:
            updated_time=time+1
            travel_time=1
        else:
            travel_time=Time_matrix[start-1][end-1][time][day]
            updated_time=int(time+travel_time)
        
        next_state_time = updated_time%24
        next_state_day = day+1 if updated_time >= 24 else day
        next_state_day = 0 if next_state_day > 6 else next_state_day
    
        return (next_state_loc,next_state_time,next_state_day,travel_time)

    
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_loc,curr_time,curr_day = state[0],state[1],state[2]
        pickup,drop = action[0],action[1]
    
        if action==(0,0):
            travel_time=1
            reward = -C
    
        else:  
            if curr_loc!=pickup:
                next_state_loc,next_state_time,next_state_day,travel_time1=self.transition(curr_loc,pickup,curr_time,curr_day,Time_matrix)
                _ ,_ ,_ ,travel_time2=self.transition(next_state_loc,drop,next_state_time,next_state_day,Time_matrix)
                reward = (R * travel_time2) - (C * (travel_time1 + travel_time2))
                travel_time=travel_time1+travel_time2
        
            else:
                _ ,_ ,_ ,travel_time=self.transition(pickup,drop,curr_time,curr_day,Time_matrix)
                reward = R * travel_time - C * (travel_time)
        return reward,travel_time




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_loc,curr_time,curr_day = state[0],state[1],state[2]
        pickup,drop = action[0],action[1]
    
        if action==(0,0):
            next_state_loc,next_state_time,next_state_day,travel_time=self.transition(curr_loc,curr_loc,curr_time,curr_day,Time_matrix)
    
        else:  
            if curr_loc!=pickup:
                next_state_loc,next_state_time,next_state_day,_ = self.transition(curr_loc,pickup,curr_time,curr_day,Time_matrix)
                next_state_loc,next_state_time,next_state_day,_ = self.transition(next_state_loc,drop,next_state_time,next_state_day,Time_matrix)
        
            else:
                next_state_loc,next_state_time,next_state_day,_ = self.transition(pickup,drop,curr_time,curr_day,Time_matrix)
        
        next_state=[next_state_loc,next_state_time,next_state_day]
        
        return(next_state)



    def reset(self):
        self.state_init = random.choice(self.state_space)
        return self.state_init
