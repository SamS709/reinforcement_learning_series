import numpy as np

class Env:
    
    def __init__(self, n: int, p: int, borders: bool, holes: bool, final_state=None, L_holes: list=[]) -> None:
        """
        Initializes the environment.

        Args:
            n (int): Number of rows.
            p (int): Number of columns.
            borders (bool): Whether to use border penalties.
            holes (bool): Whether to use holes.
            final_state (list or None): Final state position.
            L_holes (list): List of hole positions.

        Returns:
            None
        """
        self.n , self.p = n,p
        self.final_state = [n-1,p-1] if final_state == None else final_state
        self.borders = borders
        self.holes = holes  
        self.L_holes=L_holes
        self.final_reward = 1
        self.border_reward = -10
        self.hole_reward = -10
        self.generate_reward_grid()
        self.state = self.reset()

    def generate_reward_grid(self) -> None:
        """
        Generates the reward grid for the environment.

        Returns:
            None
        """
        self.reward_grid = np.array([[[0.,0.,0.,0.] for j in range(self.p)] for i in range(self.n)])
        # borders reward (negative)
        if self.borders:
            for i in range(self.n):
                self.reward_grid[i,0,2] = self.border_reward
                self.reward_grid[i,self.p-1,3] = self.border_reward
            for j in range(self.p):
                self.reward_grid[0,j,0] = self.border_reward
                self.reward_grid[self.n-1,j,1] = self.border_reward
        # holes reward (negative)        
        if self.holes:
            if len(self.L_holes) == 0:
                self.generate_random_holes()
            for hole in self.L_holes:
                self.reward_grid[hole[0]-1, hole[1],1] = self.hole_reward
                self.reward_grid[hole[0]+1, hole[1],0] = self.hole_reward
                self.reward_grid[hole[0], hole[1]-1,3] = self.hole_reward
                self.reward_grid[hole[0], hole[1]+1,2] = self.hole_reward
        # final state reward (positive)
        self.reward_grid[self.final_state[0]-1,self.final_state[1],1] = self.final_reward
        self.reward_grid[self.final_state[0],self.final_state[1]-1,3] = self.final_reward
                
    def generate_random_holes(self) -> None:
        """
        Generates random holes in the environment.

        Returns:
            None
        """
        n_holes = min(self.n,self.p)-1
        for hole in range(n_holes):
            i = np.random.randint(0,self.n)
            j = np.random.randint(0,self.p)
            self.L_holes.append([i,j])
        

    def reset(self) -> list:
        """
        Resets the environment to the initial state.

        Returns:
            list: The initial state [row, col].
        """
        self.state = [0,0]
        return self.state.copy()

    def step(self, action: int) -> tuple:
        """
        Takes an action in the environment.

        Args:
            action (int): The action to take (0=up, 1=down, 2=left, 3=right).

        Returns:
            tuple: (next_state [row, col], reward, terminated)
        """
        # determine next state
        next_state = self.state.copy()
        if action == 0 :
            next_state[0] = next_state[0] - 1 if self.state[0]>0 else self.n-1
        elif action == 1 :
            next_state[0] = next_state[0] + 1 if self.state[0]<self.n-1 else 0
        elif action == 2 :
            next_state[1] = next_state[1] - 1 if self.state[1]>0 else self.p-1
        else :
            next_state[1] = next_state[1] + 1 if self.state[1]<self.p-1 else 0
        # set reward
        reward = self.reward_grid[self.state[0], self.state[1], action]
        # determine if terminated
        terminated = False
        if next_state[0] == self.final_state[0] and next_state[1] == self.final_state[1]:
            terminated = True
        # change the current state of the environnement
        self.state = next_state.copy()
        return next_state, reward, terminated


if __name__ == "__main__":
    env = Env(n=3,p=3,holes=True,final_state=None,L_holes=[[1,1]],borders = False)
    print(env.reward_grid)
    env.reset()
