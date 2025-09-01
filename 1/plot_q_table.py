import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def plot_q_table(q_table, grid_shape, action_order=['up', 'down', 'left', 'right'], cmap='coolwarm', L_holes = [], final_path = []):
    rows, cols = grid_shape
    assert q_table.shape == (rows, cols, 4), f"Q-table shape must be (rows, cols, 4), got {q_table.shape}"

    # Use RdYlGn colormap: red for low (negative), green for high (positive)
    q_min, q_max = np.min(q_table), np.max(q_table)
    norm = plt.Normalize(q_min, q_max)
    colormap = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()

    for r in range(rows):
        for c in range(cols):
            # Draw the square for the state
            if [r,c] in final_path:
                color = 'lightgrey'
            elif [r,c] in L_holes:
                color = 'plum'
            else:
                color = 'white'
            rect = Rectangle((c, r), 1, 1, edgecolor='black', facecolor=color, lw=1)
            ax.add_patch(rect)
            # Center of the square
            cx, cy = c + 0.5, r + 0.5
            # Arrow (triangle) coordinates for each action, pointing outwards
            arrow_len = 0.45
            arrow_width = 0.3
            triangles = {
                'up':    [(cx - arrow_width/2, cy - 0.1), (cx + arrow_width/2, cy - 0.1), (cx, cy - arrow_len)],
                'down':  [(cx - arrow_width/2, cy + 0.1), (cx + arrow_width/2, cy + 0.1), (cx, cy + arrow_len)],
                'left':  [(cx - 0.1, cy - arrow_width/2), (cx - 0.1, cy + arrow_width/2), (cx - arrow_len, cy)],
                'right': [(cx + 0.1, cy - arrow_width/2), (cx + 0.1, cy + arrow_width/2), (cx + arrow_len, cy)]
            }
            for i, action in enumerate(action_order):
                q = q_table[r, c, i]
                color = colormap(norm(q))
                tri = Polygon(triangles[action], closed=True, color=color, ec='black', lw=0.5)
                ax.add_patch(tri)
                # Move Q-value text a bit towards the center
                tip = np.array(triangles[action][2])
                center = np.array([cx, cy])
                text_pos = center + 0.6 * (tip - center)  
                ax.text(text_pos[0], text_pos[1], f"{q:.2f}", ha='center', va='center', fontsize=7, color='black')
    ax.axis('off')
    plt.tight_layout()


# Example usage:
# q_table = np.random.rand(16, 4)  # for a 4x4 grid
# plot_q_table(q_table, (4, 4))
