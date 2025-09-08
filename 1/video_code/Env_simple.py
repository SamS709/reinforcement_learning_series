import numpy as np

class Env:
    
    def __init__(self, n: int, p: int) -> None:
        
        self.n , self.p = n,p
        self.final_state = [n-1,p-1]
        self.state = self.reset()
        
        

    def reset(self) -> list:

        self.state = [0,0]
        return self.state.copy()
    
    

    def step(self, action: int) -> tuple:
        
        next_state = self.state.copy()
        
        if action == 0 :
            next_state[0] = next_state[0] - 1 if self.state[0]>0 else self.n-1
        elif action == 1 :
            next_state[0] = next_state[0] + 1 if self.state[0]<self.n-1 else 0
        elif action == 2 :
            next_state[1] = next_state[1] - 1 if self.state[1]>0 else self.p-1
        else :
            next_state[1] = next_state[1] + 1 if self.state[1]<self.p-1 else 0
            
        reward = self.r(self.state,action)
        
        if next_state[0] == self.final_state[0] and next_state[1] == self.final_state[1]:
            terminated = True
        else:
            terminated = False
            
        self.state = next_state.copy()
        
        return next_state, reward, terminated
    
    
    
    def r(self,s,a) -> int: 
        
            if (s[0] == self.final_state[0] - 1 and s[1] == self.final_state[1] and a == 3
                or s[0] == self.final_state[0] and s[1] == self.final_state[1] - 1 and a == 1) :
                return 1
            
            else:
                return 0
               
                
                
    def render(self) -> None:
        print(self.state)


