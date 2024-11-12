import numpy as np
import random

# Define the maze
# 0: Free space
# 1: Wall
# 2: Food

maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 1
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # 3
    [0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 4
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],  # 5
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 6
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],  # 7
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 8
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],  # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
])

# Define positions
mouse_start = (0, 0)
food_pos = (4, 4)
cat_start = (10, 14)  # Initial position of the cat, closer to the mouse's goal

# Q-Learning parameters
alpha = 0.1        # Learning rate
gamma = 0.95       # Discount factor, increased to value future rewards more
epsilon = 1.0      # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999
num_episodes = 10000  # Increased for deeper learning
max_steps = 300

# Advantage parameters for the cat
cat_extra_move_prob = 0.3  # Probability that the cat makes an extra move

# Actions: up, down, left, right
actions = ['up', 'down', 'left', 'right']
action_dict = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}

# Initialize the Q-table
maze_height, maze_width = maze.shape
state_space_size = maze_height * maze_width * maze_height * maze_width
action_space_size = len(actions)
Q_table = np.zeros((state_space_size, action_space_size))

def state_to_index(mouse_pos, cat_pos):
    return (mouse_pos[0] * maze_width + mouse_pos[1]) * (maze_height * maze_width) + (cat_pos[0] * maze_width + cat_pos[1])

def is_valid(position):
    x, y = position
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] != 1

def get_possible_actions(position):
    possible = []
    for action, move in action_dict.items():
        new_pos = (position[0] + move[0], position[1] + move[1])
        if is_valid(new_pos):
            possible.append(action)
    return possible

def move_cat(cat_pos, mouse_pos):
    """
    Determines the best action for the cat to reduce the distance to the mouse.
    Uses the Manhattan distance to decide.
    """
    possible_actions = get_possible_actions(cat_pos)
    if not possible_actions:
        return cat_pos  # The cat cannot move

    # Calculate the current distance to the mouse
    current_distance = abs(cat_pos[0] - mouse_pos[0]) + abs(cat_pos[1] - mouse_pos[1])

    # Evaluate all possible actions and choose those that reduce the distance
    best_actions = []
    min_distance = current_distance
    for action in possible_actions:
        move = action_dict[action]
        new_pos = (cat_pos[0] + move[0], cat_pos[1] + move[1])
        distance = abs(new_pos[0] - mouse_pos[0]) + abs(new_pos[1] - mouse_pos[1])
        if distance < min_distance:
            min_distance = distance
            best_actions = [action]
        elif distance == min_distance:
            best_actions.append(action)

    if best_actions:
        chosen_action = random.choice(best_actions)
        move = action_dict[chosen_action]
        return (cat_pos[0] + move[0], cat_pos[1] + move[1])
    else:
        # No actions reduce the distance, move randomly
        chosen_action = random.choice(possible_actions)
        move = action_dict[chosen_action]
        return (cat_pos[0] + move[0], cat_pos[1] + move[1])

for episode in range(num_episodes):
    mouse_pos = mouse_start
    cat_pos = cat_start
    total_reward = 0
    done = False  # Ensure 'done' is defined at the start of each episode

    for step in range(max_steps):
        state_idx = state_to_index(mouse_pos, cat_pos)

        # Action selection using ε-greedy
        if random.uniform(0,1) < epsilon:
            possible = get_possible_actions(mouse_pos)
            if possible:
                action = random.choice(possible)
            else:
                action = random.randint(0, action_space_size -1)
        else:
            action = np.argmax(Q_table[state_idx])

        # Save previous positions
        prev_mouse_pos = mouse_pos
        prev_cat_pos = cat_pos

        # Take the mouse's action
        move = action_dict[action]
        new_mouse_pos = (mouse_pos[0] + move[0], mouse_pos[1] + move[1])

        if not is_valid(new_mouse_pos):
            reward = -10  # Invalid move
            new_mouse_pos = mouse_pos  # Stay in place
            # 'done' remains False
        elif new_mouse_pos == food_pos:
            reward = 100  # Reaches the food
            done = True
        else:
            # Additional penalty for proximity to the cat
            distance_to_cat = abs(new_mouse_pos[0] - cat_pos[0]) + abs(new_mouse_pos[1] - cat_pos[1])
            if distance_to_cat <= 2:
                reward = -5  # Penalty for being close to the cat
            else:
                reward = -1  # Normal move

        # Move the cat with speed advantage
        # First, move once
        new_cat_pos = move_cat(cat_pos, new_mouse_pos)

        # Check for capture after the first move
        captured = False
        if new_cat_pos == new_mouse_pos:
            captured = True
        elif new_cat_pos == prev_mouse_pos and new_mouse_pos == prev_cat_pos:
            captured = True

        # If captured, assign reward and finish
        if captured:
            reward = -100  # Captured by the cat
            done = True
        else:
            # Decide if the cat makes an extra move
            if random.uniform(0,1) < cat_extra_move_prob:
                temp_cat_pos = new_cat_pos
                temp_prev_cat_pos = new_cat_pos
                temp_prev_mouse_pos = new_mouse_pos

                # Extra move
                new_cat_pos = move_cat(temp_cat_pos, new_mouse_pos)

                # Check for capture after the extra move
                if new_cat_pos == new_mouse_pos:
                    captured = True
                elif new_cat_pos == temp_prev_mouse_pos and new_mouse_pos == temp_prev_cat_pos:
                    captured = True

                if captured:
                    reward = -100  # Captured by the cat
                    done = True

        # Get the index of the new state for the mouse and the cat
        new_state_idx = state_to_index(new_mouse_pos, new_cat_pos)

        # Q-learning update
        Q_table[state_idx, action] += alpha * (reward + gamma * np.max(Q_table[new_state_idx]) - Q_table[state_idx, action])

        # Update positions
        mouse_pos = new_mouse_pos
        cat_pos = new_cat_pos
        total_reward += reward

        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print progress every 1000 episodes
    if (episode+1) % 1000 == 0:
        print(f"Episode {episode+1}: Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# Test the trained mouse
def print_maze(mouse_position, cat_position):
    display = maze.copy().astype(object)
    # Reset the display
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if maze[i, j] == 1:
                display[i, j] = '#'
            elif maze[i, j] == 2:
                display[i, j] = 'F'
            else:
                display[i, j] = '.'
    # Place the mouse and the cat
    mx, my = mouse_position
    cx, cy = cat_position
    # Avoid having the mouse and the cat in the same position in the visualization
    if (mx, my) == (cx, cy):
        display[mx, my] = 'X'  # Indicates that the cat has captured the mouse
    else:
        display[mx, my] = 'M'
        display[cx, cy] = 'C'
    # Print the maze
    for row in display:
        print(' '.join(row))
    print()

# Reset parameters for testing
mouse_pos = mouse_start
cat_pos = cat_start
epsilon = 0  # No exploration

print("Trained Mouse's Path:")
for step in range(max_steps):
    state_idx = state_to_index(mouse_pos, cat_pos)
    possible = get_possible_actions(mouse_pos)
    if possible:
        action = np.argmax(Q_table[state_idx])
    else:
        action = random.randint(0, action_space_size -1)
    move = action_dict[action]
    new_mouse_pos = (mouse_pos[0] + move[0], mouse_pos[1] + move[1])

    if not is_valid(new_mouse_pos):
        new_mouse_pos = mouse_pos  # Stay in place

    # Save previous positions to detect crossings
    prev_mouse_pos = mouse_pos
    prev_cat_pos = cat_pos

    # Take the mouse's action
    mouse_pos = new_mouse_pos

    # Move the cat with speed advantage
    # First, move once
    new_cat_pos = move_cat(cat_pos, new_mouse_pos)

    # Check for capture after the first move
    captured = False
    if new_cat_pos == new_mouse_pos:
        captured = True
    elif new_cat_pos == prev_mouse_pos and new_mouse_pos == prev_cat_pos:
        captured = True

    # If not captured, decide if the cat makes an extra move
    if not captured and random.uniform(0,1) < cat_extra_move_prob:
        temp_cat_pos = new_cat_pos
        temp_prev_cat_pos = new_cat_pos
        temp_prev_mouse_pos = new_mouse_pos

        # Extra move
        new_cat_pos = move_cat(temp_cat_pos, new_mouse_pos)

        # Check for capture after the extra move
        if new_cat_pos == new_mouse_pos:
            captured = True
        elif new_cat_pos == temp_prev_mouse_pos and new_mouse_pos == temp_prev_cat_pos:
            captured = True

    # Update the cat's positions
    cat_pos = new_cat_pos

    print_maze(new_mouse_pos, new_cat_pos)

    # Check end conditions
    if new_mouse_pos == food_pos:
        print("The mouse has reached the food!")
        break
    if captured:
        print("The mouse was captured by the cat!")
        break
else:
    print("The mouse did not reach the food because the cat kept it away!")