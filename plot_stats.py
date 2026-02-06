import matplotlib.pyplot as plt
import numpy as np

# Plot with rolling average
def plot_training_statistics(avg_scores, avg_moves, epsilons, view_type,lr, stats):
      episodes = np.arange(0, len(avg_scores) * 50, 50)

      # Create figure with 3 subplots
      fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

      # Plot 1: Average Score
      ax1.plot(episodes, avg_scores, 'b-', linewidth=2)
      ax1.set_xlabel('Episode')
      ax1.set_ylabel('Average Score')
      ax1.set_title('Average Score over Training')
      ax1.grid(True, alpha=0.3)

      # Plot 2: Average Moves
      ax2.plot(episodes, avg_moves, 'g-', linewidth=2)
      ax2.set_xlabel('Episode')
      ax2.set_ylabel('Average Moves')
      ax2.set_title('Average Moves per Game')
      ax2.grid(True, alpha=0.3)

      # Plot 3: Epsilon
      ax3.plot(episodes, epsilons, 'r-', linewidth=2)
      ax3.set_xlabel('Episode')
      ax3.set_ylabel('Epsilon')
      ax3.set_title('Exploration Rate (Epsilon)')
      ax3.grid(True, alpha=0.3)

      plt.tight_layout()
      plt.savefig(f'training_results_{view_type}_{stats}_lr{lr}.png', dpi=300)  # Save to file
      plt.show()          