from snake_game import SnakeGame
from deep_q_learning_torch import DQN, DDQN
from train_test_make_nnet import test_model, train_model, build_model, just_play
from plot_stats import plot_training_statistics


#Directional with epsilon decaying to 0.03 is for training
if __name__ == '__main__':
      view_type = 'directional'
      episodes_ = 3_000
      _batch_size = 256
      _memory_size = 8_192


      env = SnakeGame(ren=True, speed=0)

      shape = env.get_state(view_type)
      input_dim = shape.shape[0] * shape.shape[1] if len(shape.shape) > 1 else shape.shape[0]
      output_dim = 3  # [straight, left, right]

      model = build_model(input_dim, output_dim)
      target_model = build_model(input_dim, output_dim)
      
      agent = DQN(input_dim, output_dim, model,target_model, learning_rate=0.001, batch_size=_batch_size, memory_size=_memory_size)
      
      
      play = input("Let the agent just play a game? (y/n)")
      if play == 'y':
            agent.load("Snake_directional_type_35.88_episodes_3000_[bat=256,mem=8_192]","round2")
            just_play(100,agent, SnakeGame(ren=True, speed=0, test=True), view_type)
      
      
      x = input("First train the model? (y/n)") 
      if x == 'y':      
            (avg_score, avg_moves, epsilon) = train_model(agent, env, view_type, num_episodes=episodes_)
            plot_training_statistics(avg_score, avg_moves, epsilon, view_type=view_type, lr = 0.001, stats=f'[bat={_batch_size},mem={_memory_size}]')
            
            test_model(agent, iterations=100, env=SnakeGame(ren=True, speed=0, test=True), view_type=view_type, name_agent='Snake', episodes=episodes_)
            
      else:
            #write the name of the agent to be trained another round
            agent.load("Snake_directional_type_33.66_episodes_3000_[bat=256,mem=8_192]","round1")
            (avg_score, avg_moves, epsilon) = train_model(agent, env, view_type, num_episodes=1_000, round2=True)

            plot_training_statistics(avg_score, avg_moves, epsilon, view_type=view_type, lr = 0.001, stats=f'[bat={_batch_size},mem={_memory_size}]_round2')

            test_model(agent, iterations=100, env=SnakeGame(ren=True, speed=0, test=True), view_type=view_type, name_agent='Snake', episodes=episodes_)
