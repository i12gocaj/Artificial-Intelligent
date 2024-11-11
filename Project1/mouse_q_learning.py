import numpy as np
import random

# Definir el laberinto
# 0: Espacio libre
# 1: Pared
# 2: Comida

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

# Definir posiciones
mouse_start = (0, 0)
food_pos = (4, 4)
cat_start = (10, 14)  # Posición inicial del gato, más cercana al objetivo del ratón

# Parámetros de Q-Learning
alpha = 0.1        # Tasa de aprendizaje
gamma = 0.95       # Factor de descuento, incrementado para valorar más las recompensas futuras
epsilon = 1.0      # Tasa de exploración
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 10000  # Aumentado para un aprendizaje más profundo
max_steps = 300

# Parámetros de ventaja para el gato
cat_extra_move_prob = 0.3  # Probabilidad de que el gato realice un movimiento extra

# Acciones: arriba, abajo, izquierda, derecha
actions = ['arriba', 'abajo', 'izquierda', 'derecha']
action_dict = {
    0: (-1, 0),  # Arriba
    1: (1, 0),   # Abajo
    2: (0, -1),  # Izquierda
    3: (0, 1),   # Derecha
}

# Inicializar la tabla Q
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
    Determina la mejor acción para el gato para reducir la distancia al ratón.
    Utiliza la distancia de Manhattan para decidir.
    """
    possible_actions = get_possible_actions(cat_pos)
    if not possible_actions:
        return cat_pos  # El gato no puede moverse

    # Calcular la distancia actual al ratón
    current_distance = abs(cat_pos[0] - mouse_pos[0]) + abs(cat_pos[1] - mouse_pos[1])

    # Evaluar todas las acciones posibles y elegir las que reduzcan la distancia
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
        # No hay acciones que reduzcan la distancia, moverse aleatoriamente
        chosen_action = random.choice(possible_actions)
        move = action_dict[chosen_action]
        return (cat_pos[0] + move[0], cat_pos[1] + move[1])

for episode in range(num_episodes):
    mouse_pos = mouse_start
    cat_pos = cat_start
    total_reward = 0
    done = False  # Asegurar que 'done' está definido al inicio de cada episodio

    for step in range(max_steps):
        state_idx = state_to_index(mouse_pos, cat_pos)

        # Selección de acción usando ε-greedy
        if random.uniform(0,1) < epsilon:
            possible = get_possible_actions(mouse_pos)
            if possible:
                action = random.choice(possible)
            else:
                action = random.randint(0, action_space_size -1)
        else:
            action = np.argmax(Q_table[state_idx])

        # Guardar posiciones anteriores
        prev_mouse_pos = mouse_pos
        prev_cat_pos = cat_pos

        # Tomar acción del ratón
        move = action_dict[action]
        new_mouse_pos = (mouse_pos[0] + move[0], mouse_pos[1] + move[1])

        if not is_valid(new_mouse_pos):
            reward = -10  # Movimiento inválido
            new_mouse_pos = mouse_pos  # Quedarse en el lugar
            # 'done' se mantiene en False
        elif new_mouse_pos == food_pos:
            reward = 100  # Alcanza la comida
            done = True
        else:
            # Penalización adicional por proximidad al gato
            distance_to_cat = abs(new_mouse_pos[0] - cat_pos[0]) + abs(new_mouse_pos[1] - cat_pos[1])
            if distance_to_cat <= 2:
                reward = -5  # Penalización por estar cerca del gato
            else:
                reward = -1  # Movimiento normal

        # Mover al gato con ventaja de velocidad
        # Primero, mover una vez
        new_cat_pos = move_cat(cat_pos, new_mouse_pos)

        # Verificar captura después del primer movimiento
        captured = False
        if new_cat_pos == new_mouse_pos:
            captured = True
        elif new_cat_pos == prev_mouse_pos and new_mouse_pos == prev_cat_pos:
            captured = True

        # Si capturado, asignar recompensa y finalizar
        if captured:
            reward = -100  # Capturado por el gato
            done = True
        else:
            # Decidir si el gato hace un movimiento extra
            if random.uniform(0,1) < cat_extra_move_prob:
                temp_cat_pos = new_cat_pos
                temp_prev_cat_pos = new_cat_pos
                temp_prev_mouse_pos = new_mouse_pos

                # Movimiento extra
                new_cat_pos = move_cat(temp_cat_pos, new_mouse_pos)

                # Verificar captura después del movimiento extra
                if new_cat_pos == new_mouse_pos:
                    captured = True
                elif new_cat_pos == temp_prev_mouse_pos and new_mouse_pos == temp_prev_cat_pos:
                    captured = True

                if captured:
                    reward = -100  # Capturado por el gato
                    done = True

        # Obtener el índice del nuevo estado del ratón y el gato
        new_state_idx = state_to_index(new_mouse_pos, new_cat_pos)

        # Actualización Q-learning
        Q_table[state_idx, action] += alpha * (reward + gamma * np.max(Q_table[new_state_idx]) - Q_table[state_idx, action])

        # Actualizar posiciones
        mouse_pos = new_mouse_pos
        cat_pos = new_cat_pos
        total_reward += reward

        if done:
            break

    # Decaimiento de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Imprimir progreso cada 1000 episodios
    if (episode+1) % 1000 == 0:
        print(f"Episodio {episode+1}: Recompensa Total: {total_reward}, Epsilon: {epsilon:.4f}")

# Probar el ratón entrenado
def print_maze(mouse_position, cat_position):
    display = maze.copy().astype(object)
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
    # Evitar que el ratón y el gato estén en la misma posición en la visualización
    if (mx, my) == (cx, cy):
        display[mx, my] = 'X'  # Indica que el gato ha capturado al ratón
    else:
        display[mx, my] = 'M'
        display[cx, cy] = 'C'
    # Imprimir el laberinto
    for row in display:
        print(' '.join(row))
    print()

# Restablecer parámetros para prueba
mouse_pos = mouse_start
cat_pos = cat_start
epsilon = 0  # Sin exploración

print("Camino del Ratón Entrenado:")
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
        new_mouse_pos = mouse_pos  # Quedarse en el lugar

    # Guardar posiciones anteriores para detectar cruces
    prev_mouse_pos = mouse_pos
    prev_cat_pos = cat_pos

    # Tomar acción del ratón
    mouse_pos = new_mouse_pos

    # Mover al gato con ventaja de velocidad
    # Primero, mover una vez
    new_cat_pos = move_cat(cat_pos, new_mouse_pos)

    # Verificar captura después del primer movimiento
    captured = False
    if new_cat_pos == new_mouse_pos:
        captured = True
    elif new_cat_pos == prev_mouse_pos and new_mouse_pos == prev_cat_pos:
        captured = True

    # Si no capturado, decidir si el gato hace un movimiento extra
    if not captured and random.uniform(0,1) < cat_extra_move_prob:
        temp_cat_pos = new_cat_pos
        temp_prev_cat_pos = new_cat_pos
        temp_prev_mouse_pos = new_mouse_pos

        # Movimiento extra
        new_cat_pos = move_cat(temp_cat_pos, new_mouse_pos)

        # Verificar captura después del movimiento extra
        if new_cat_pos == new_mouse_pos:
            captured = True
        elif new_cat_pos == temp_prev_mouse_pos and new_mouse_pos == temp_prev_cat_pos:
            captured = True

    # Actualizar posiciones del gato
    cat_pos = new_cat_pos

    print_maze(new_mouse_pos, new_cat_pos)

    # Verificar condiciones de fin
    if new_mouse_pos == food_pos:
        print("¡El ratón ha llegado a la comida!")
        break
    if captured:
        print("¡El ratón fue capturado por el gato!")
        break