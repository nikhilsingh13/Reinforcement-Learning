# Reinforcement-Learning
 Understanding and working on Reinforcement Learning

## About the folders:
1. Multi-armed bandits
	>	Implementation of Multi-armed bandits algorithm for 10 armed test-bed using :    
    1. greedy algorithm
    2. epsilon-greedy algorithm
    3. UCB

2. Monte Carlo Simulation
  > Implementation of `Monte Carlo first visit` algorithm for Grid-world problem.

3. Implementation of Q-learning on the game of Pacman
  > - The code of Pacman game is downloaded from the website of [UC Berkeley](http://ai.berkeley.edu/reinforcement.html).     
  > - The implementation of Q learning algorithm is in the file `qlearningagents.py`.

  - To run the game, download all files including layout folder from the repo, maintaining the similar folder structure as on repository.
  - Open terminal/cmd, change the directory to the same directory where the `pacman.py` file exists.
  - Run below mentioned commands to see either agent playing the game or You.
    - _python pacman.py_ - runs the game and can be played by You
    - _python pacman.py --help_ - shows all the options available
    - Q-learning agent will play the game in different map layouts as mentioned at the end of the commands.
    > python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid    
      python pacman.py -p QLearnAgent -x 2000 -n 2010 -l testClassic     
      python pacman.py -p QLearnAgent -x 2000 -n 2010 -l trappedClassic     
      python pacman.py -p QLearnAgent -x 2000 -n 2010 -l minimaxClassic     
      python pacman.py -p QLearnAgent -x 2000 -n 2010 -l mediumGrid      



Python version: 3.7.3
