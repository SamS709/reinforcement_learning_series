import numpy as np
import time
import matplotlib.pyplot as plt
from plot_q_table import plot_q_table

class Trainer:

    def __init__(self,n,p,borders = False,holes = True, final_state = None,L_holes = [], learning_rate = 0.1, discount_factor = 0.98):

        from Env import Env
        self.env = Env(n,p,borders,holes,final_state = final_state,L_holes = L_holes)
        self.L_holes = L_holes
        self.q_table = np.array([[[0.,0.,0.,0.] for j in range(p)] for i in range(n)])
        self.epsilon = 0.5
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.step_reward = 0.0
        self.n = n
        self.p = p

    def epsilon_greedy(self, state):

        if np.random.rand()<self.epsilon: 
            random_action = np.random.randint(0,4)
            return random_action
        else:
            actions = self.q_table[state[0]][state[1]]
            best_actions = np.argwhere(actions == np.amax(actions))
            return np.random.choice(best_actions.flatten())

    def update_q_table(self,state,next_state,action,reward,terminated):

        max_next_q_val = max(self.q_table[next_state[0]][next_state[1]])
        curr_q = self.q_table[state[0]][state[1]][action]
        next_q_val = curr_q + self.learning_rate * (reward + (1-terminated)*self.discount_factor * max_next_q_val - curr_q)
        self.q_table[state[0]][state[1]][action] = next_q_val

    def best_path(self):
        path = np.array([[0 for j in range(self.p)]for i in range(self.n)])
        for hole in self.L_holes:
            path[hole[0],hole[1]] = -1 # holes represented by -1 in the finale printed grid
        state = self.env.reset()
        path[state[0],state[1]] = 1
        terminated = False
        k = 0
        while not terminated and k<=20:
            k+=1
            actions = self.q_table[state[0]][state[1]]
            best_action = np.argmax(actions)
            next_state, reward, terminated = self.env.step(best_action)
            path[next_state[0],next_state[1]]=1
            state = next_state
        return path



    def train(self,n_epochs,show_final_path = False, show_table =False, show_graphs = False):

        rewards = []
        path_lengths = []

        for epoch in range(n_epochs):
            reward, path_length = self.play_one_step()
            rewards.append(reward)
            path_lengths.append(path_length)
            rewards_arr = np.array(rewards)
            path_lengths_arr = np.array(path_lengths)
            mean_reward = np.mean(rewards_arr[len(rewards)-10:])
            mean_path_length = np.mean(path_lengths_arr[len(path_lengths)-10:])
            
            # Clear screen and move cursor to top
            print("\033[2J\033[H", end="")
            print(f"Epoch: {epoch+1}/{n_epochs} | Mean reward: {mean_reward:.3f} | Mean path length: {mean_path_length:.3f}")
            if show_table:
                print("Q-table:")
                np.set_printoptions(precision=3, suppress=True)
                print(self.q_table)
                print("="*80)
                time.sleep(0.001)

        if show_final_path:
            best_path = self.best_path()
            print(best_path)
            
        if show_graphs:
            
            chunk_size = 100
            mean_rewards = [np.mean(rewards[i:i+chunk_size]) for i in range(0, len(rewards), chunk_size)]
            mean_path_lengths = [np.mean(path_lengths[i:i+chunk_size]) for i in range(0, len(path_lengths), chunk_size)]
            plt.figure("Rewards")
            plt.plot(mean_rewards, label="Reward")
            plt.figure("Path lengths")
            plt.plot(mean_path_lengths, label="Path Length")
            plt.xlabel("Epoch")
            plt.legend()
            plot_q_table(self.q_table,(self.n,self.p))
            plt.show()
            
            
    def play_one_step(self):

        terminated = False
        state = self.env.reset()
    
        reward = 0
        path_length = 0

        while not terminated:
            action = self.epsilon_greedy(state)
            next_state, reward, terminated = self.env.step(action)
            self.update_q_table(state,next_state,action,reward,terminated)
            path_length+=1
            reward+=reward
            state = next_state

        return reward, path_length


if __name__=="__main__":
    L_wholes = [[1,0],[1,1]]
    trainer = Trainer(5,5,borders = True,holes=True,final_state=None,L_holes=L_wholes)
    trainer.train(10000,show_final_path=True,show_table = False, show_graphs = True)