import numpy as np
import random

maze = np.array([
    [ 0,  0,  0, -1],
    [ 0, -1,  0, -1],
    [ 0,  0,  0,  0],
    [-1,  0, -1,  1]
])

rows, cols = maze.shape
start_state = (0, 0)
goal_state = (3, 3)

actions = [0, 1, 2, 3]
action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

q_table = np.zeros((rows, cols, len(actions)))

alpha = 0.1      
gamma = 0.9      
epsilon = 0.2    
episodes = 500   

def step(state, action):
    r, c = state
    if action == 0:  
        r = max(r - 1, 0)
    elif action == 1: 
        r = min(r + 1, rows - 1)
    elif action == 2:
        c = max(c - 1, 0)
    elif action == 3:
        c = min(c + 1, cols - 1)
        
    next_state = (r, c)
    
    if next_state == goal_state:
        reward = 100
        done = True
    elif maze[r, c] == -1: 
        reward = -10
        next_state = state 
        done = False
    else:
        reward = -1 
        done = False
        
    return next_state, reward, done

for episode in range(episodes):
    state = start_state
    done = False
    
    while not done:
        r, c = state
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions) 
        else:
            action = np.argmax(q_table[r, c]) 
            
        next_state, reward, done = step(state, action)
        nr, nc = next_state
        
        old_q = q_table[r, c, action]
        max_future_q = np.max(q_table[nr, nc])
        q_table[r, c, action] = old_q + alpha * (reward + gamma * max_future_q - old_q)
        
        state = next_state

print("Training Completed!\n")

state = start_state
path = [state]
moves = []

while state != goal_state:
    r, c = state
    best_action = np.argmax(q_table[r, c])
    moves.append(action_names[best_action])
    state, _, _ = step(state, best_action)
    path.append(state)

print("Optimal Path (Coordinates):", " -> ".join([str(p) for p in path]))
print("Sequence of Actions       :", " -> ".join(moves))

visual_grid = np.full((rows, cols), '.', dtype=str)
for r in range(rows):
    for c in range(cols):
        if maze[r, c] == -1:
            visual_grid[r, c] = '#'
for (r, c) in path:
    visual_grid[r, c] = '*'
visual_grid[start_state] = 'S'
visual_grid[goal_state] = 'G'

print("\nMaze Solution ('S'=Start, 'G'=Goal, '*'=Path, '#'=Wall):")
for row in visual_grid:
    print("  ".join(row))