import numpy as np
import random

# Definir el laberinto
# 0: Espacio libre
# 1: Pared
# 2: Comida
# 3: Ratón
# 4: Gato

maze = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 2, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

# Definir posiciones
mouse_start = (0, 0)
food_pos = (3, 3)
cat_route = [
    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), 
    (4,6), (3,6), (2,6), (1,6), (0,6),
    (0,5), (0,4), (0,3), (0,2), (0,1), (0,0), 
    (1,0), (2,0), (3,0), (4,0), (5,0)
]  # Recorrido en bucle alrededor del laberinto

# Parámetros de Q-Learning
alpha = 0.1        # Tasa de aprendizaje
gamma = 0.9        # Factor de descuento
epsilon = 1.0      # Tasa de exploración
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000
max_steps = 100

# Acciones: arriba, abajo, izquierda, derecha
actions = ['arriba', 'abajo', 'izquierda', 'derecha']
action_dict = {
    0: (-1, 0),  # Arriba
    1: (1, 0),   # Abajo
    2: (0, -1),  # Izquierda
    3: (0, 1),   # Derecha
}

# Inicializar la tabla Q
state_space_size = maze.shape[0] * maze.shape[1]
action_space_size = len(actions)
Q_table = np.zeros((state_space_size, action_space_size))

def state_to_index(state):
    return state[0] * maze.shape[1] + state[1]

def is_valid(position):
    x, y = position
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] != 1

def get_cat_position(step):
    return cat_route[step % len(cat_route)]

for episode in range(num_episodes):
    mouse_pos = mouse_start
    total_reward = 0
    done = False  # Asegurarse de que 'done' está definido al inicio de cada episodio

    for step in range(max_steps):
        state_idx = state_to_index(mouse_pos)

        # Selección de acción usando ε-greedy
        if random.uniform(0,1) < epsilon:
            action = random.randint(0, action_space_size -1)
        else:
            action = np.argmax(Q_table[state_idx])

        # Tomar acción
        move = action_dict[action]
        new_mouse_pos = (mouse_pos[0] + move[0], mouse_pos[1] + move[1])

        if not is_valid(new_mouse_pos):
            reward = -10  # Movimiento inválido
            new_mouse_pos = mouse_pos  # Quedarse en el lugar
            done = False  # No termina el episodio
        elif new_mouse_pos == food_pos:
            reward = 100  # Alcanza la comida
            done = True
        else:
            reward = -1  # Movimiento normal
            done = False

        # Actualizar posición del gato
        cat_pos = get_cat_position(step)
        if new_mouse_pos == cat_pos:
            reward = -100  # Capturado por el gato
            done = True

        # Obtener el índice del nuevo estado
        new_state_idx = state_to_index(new_mouse_pos)

        # Actualización Q-learning
        Q_table[state_idx, action] = Q_table[state_idx, action] + alpha * (reward + gamma * np.max(Q_table[new_state_idx]) - Q_table[state_idx, action])

        mouse_pos = new_mouse_pos
        total_reward += reward

        if done:
            break

    # Decaimiento de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode+1) % 100 == 0:
        print(f"Episodio {episode+1}: Recompensa Total: {total_reward}, Epsilon: {epsilon:.4f}")

# Probar el ratón entrenado
def print_maze(mouse_position, cat_position):
    display = maze.copy()
    display = display.astype(object)  # Para poder asignar diferentes tipos
    # Reiniciar el display
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if maze[i, j] == 1:
                display[i, j] = '#'
            elif maze[i, j] == 2:
                display[i, j] = 'F'
            else:
                display[i, j] = '.'
    # Colocar al ratón y al gato
    mx, my = mouse_position
    cx, cy = cat_position
    display[mx, my] = 'M'
    display[cx, cy] = 'C'
    # Imprimir el laberinto
    for row in display:
        print(' '.join(row))
    print()

# Restablecer parámetros para prueba
mouse_pos = mouse_start
epsilon = 0  # Sin exploración

print("Camino del Ratón Entrenado:")
for step in range(max_steps):
    state_idx = state_to_index(mouse_pos)
    action = np.argmax(Q_table[state_idx])
    move = action_dict[action]
    new_mouse_pos = (mouse_pos[0] + move[0], mouse_pos[1] + move[1])

    if not is_valid(new_mouse_pos):
        new_mouse_pos = mouse_pos  # Quedarse en el lugar

    cat_pos = get_cat_position(step)
    print_maze(new_mouse_pos, cat_pos)

    if new_mouse_pos == food_pos:
        print("¡El ratón ha llegado a la comida!")
        break
    if new_mouse_pos == cat_pos:
        print("¡El ratón fue capturado por el gato!")
        break

    mouse_pos = new_mouse_pos