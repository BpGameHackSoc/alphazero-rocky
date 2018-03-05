# Rocky (AlphaZero at home)

Rocky is a simple implementation of AlphaZero. The engine is flexible for any perfect-information, 2-player zero-sum games, such as tic-tac-toe, Gomoku, 3d-Gomoku, Connect-4 or Othello. The code was written in python and is under development.

## Goals

The goal of this project is to understand the algorithm of AlphaZero and go beyond, finding possible improvements. Although DeepMind's program's performance is mind-blowing, it still requires thousands of TPUs to be trained and is not convenient for home use yet. Maybe we'll find a way to speed up and simplify the training. Fingers crossed.

## Possible improvements on AlphaZero

### The view of the board

[source](http://teleported.in/posts/analysing-alphago/)

<img src="http://teleported.in/post_imgs/04-alphago.jpg" alt="go_board" width="250" height="250" align="right" />

As described in this [paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf), DeepMind used an entire image channel to tell the neural network who's turn it is in the given state.
> The final feature plane, C, represents the colour to play and has a constant value of either 1 if black
> is to play or 0 if white is to play.

So normally the neural network takes 2N+1 inputs. In the case of a single chess state, it would be represented by 13 layers: 6 for each white piece, 6 for each black one and 1 for the player to turn.

But we believe it's not exactly the right way to look at the board. When a player looks at the board, she doesn't see black or white pieces. She sees pieces owned by her, and pieces owned by the opponent player. Therefore in Rocky, the feature plane C mentioned in the quote was removed. Instead, we used the first N layers to represent the pieces of the player who has the right to move, and the last N layer to the opponent pieces.

### Exploration hurts for ground truth value

In the last decades, temporal difference learning has become quite popular. Nevertheless, for some reason DeepMind did not use it, but simply fed back the outcome of the game into both the ending and opening positions. In other words, if white side won a game and the final result was interpreted as -1, even the very first state was marked with the -1 label during the training of the neural net.

We suspect a minor but possible issue: exploration moves. What if we already have a quite perfect evaluation function, but we keep forcing our engine to make exploration moves? The training set will become noisy and we could hardly improve the program if it already has a considerably strong level.

We would like to propose a bit different approach. At first, when the engine is dumpy, it's okay to make feedback of the final outcome everywhere, because the computer needs some initial knowledge. But, given that we know exactly how likely it is that the chosen move is beneficial, temporal difference learning can help us to cut off exploration moves, so the original knowledge wouldn't get get corrupted.

Let me tell you an example. The neural net predicted the root node's value to be 0.7 (scores go from -1 to 1). We are in the root node and we have two possible actions, A and B. The MCTS says we have visited action A 90 times, while action B was only visited 10 times, thus A has 0.9 chance to be chosen while B only has 0.1. Assuming we choose B, and finally, we lose the game, it's not quite right to say this position's value should be -1. Instead, because it's given the chosen move had 10% probability to be the best one, we should rather scale the new value as: 0.9 * [0.7] + 0.1 * [-1].

Because this method would leave the opening positions empty for a very long time when we kick of the training, it might be a good idea to use a temperature variable to control how much the engine is allowed to learn from exploration moves. 

### Evolutionary algorithm

Although DeepMind had the opportunity to run 1600 simulations in about 0.4s, even for the simplest game it's not really possible at home. But, we have a strong hope that maybe the key of the algorithm is not the MCTS's supervision on the policy network, but the evulational steps how AlphaZero grows by self-play. Assuming we're playing a game with a finite number of states, the perfect evaluation function must exist, thus, maybe it's possible two get a closer estimate by only relying on decisions carried out by the polcy network.

In summary, we'd like to try out what if the MCTS is removed and we continue with the policy network only. The idea would be to learn 0 or 1 for all actions chosen during a game, but with the application of temporal difference learning, the same way as described in the section above. When the self-play is done, we would use not only two but dozens of agents to challenge the current best model.


## Other differences

 - This engine uses ___ CPU and ____ GPU, while AlphaZero used 5000 TPUs
 - Everything is executed on one core, no parallelization was done
 - To speed up the learning, the terminal nodes in the MCTS return with the true value (-1, 0, 1)
 
## References

- Silver, David, et al. "Mastering the Game of Go without Human Knowledge." Nature 550.7676 (2017): 354.
