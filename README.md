# Rocky (AlphaZero and possible improvements)

Rocky is a simple implementation of AlphaZero. The engine is flexible for any perfect-information, 2-player zero-sum games, such as tic-tac-toe, gomoku, 3d-gomoku, connect-4 or othello. The code was written in python and is still under development.

## Goals

The goals of this project is to understand the algorithm of AlphaZero and go beyond, finding possible improvements. Although DeepMind's program's performance is mind-blowing, it still requires thousands of TPUs to be trained and is not convenient for home use yet. Maybe we'll find a way to speed the training up and not to get stuck in a local maxima at the same time. Fingers crossed.

## Possible improvements on AlphaZero

### The view of the board

![alt convolutional network](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)
[source](https://www.tastehit.com/blog/google-deepmind-alphago-how-it-works/)

As it is described in their [paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf), DeepMind used an entire image channel to tell the neural network who's turn it is in the given state.
> The final feature plane, C, represents the colour to play, and has a constant value of either 1 if black
> is to play or 0 if white is to play.

So normally the neural network takes 2N+1 inputs. In the case of a single chess state, it would be represented by 13 layers: 6 for each white piece, 6 for each black one and 1 for the player to turn.

As a junior data scientists we experienced when we're about doing regression, categorical columns can be treated in two different ways: dummy one-hot features or training seperate models for each category. In case of a game, the 'turn' feature can be considered as a categorical column. DeepMind chose the one-hot feature represenation, but we believe it's not exactly the right way to look at the problem.

When a player normally looks at the board, she doesn't see black or white pieces. She sees pieces owned by her, and pieces owned by the opponent player. Therefore in Rocky, the feature plane C mentioned in the quote is removed. Instead we use the first N layers always for the player who has the right to move, and the second N layer who's in passive position. 

### Vanishing probabilities

DeepMind used the following formula to guide AlphaZero in MCTS's selection phrase:
![alt math](https://imgur.com/jDvuQwY)

[If you have troubles with understanding the formula, please find detailed explanation in their [paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf).]
Now, let's assume we train our engine at home, so we cannot afford 1600 simulations as AlphaZero can (even 1600 can be dangerous in the case of Go). If we are not careful enough, a low number of simulation can lead to unvisited children node at the root. In itself it wouldn't be a problem, but we train our policy network to return with the same result as the MCTS does. Consequently, accientaly a child node's probability is set to 0 and it might be possible a trained network will never again execute this node again.

We can either set up lower bounds for the probabilities, or maybe MCTS is just simply not the perfect supervisor to train the policy side of our network.

### Exploration hurts for ground truth value

In the last decades temporal difference learning has become quite popular. Nevertheless, for whatever reason DeepMind did not used it, but simply fed back the outcome of the game into both the ending and opening positions. In other words, if white side won a game and the final result was interpreted as -1, even the very first state was marked with the -1 label during the training of the neural net.

The problem with this approach is simple: exploration moves. What if we already have a quite perfect evaluation function, but we keep forcing our engine to make exploration moves sometimes? The training set will become noisy and we could hardly improve the program if it already has a considerably strong level.

We would like to propose a bit different approach. At first, when the engine is dump, it's okay to make feed back the final outcome everywhere, because the computer needs some initial knowledge. But, given that we know exactly how likely it is that the chosen move is beneficial, temporal difference learning can help us to cut off exploration moves, so the original knowledge wouldn't get hurt so much.

Let me tell you an example. The neural net predicted the root node's value to be 0.7 (scores go from -1 to 1). We are in the root node and we have two possible actions, A and B. The MCTS says we have visited action A 90 times, while action B was only visited 10 times, thus A has 0.9 chance to be chosen while B only has 0.1. Assuming we choose B, and finally we lose the game, it's not quite right to say this position's value should be -1. Instead, because it's given the chosen move had 10% probability to be the best one, we should rather scale the new value as: 0.9 * V<sub>t</sub> + 0.1 * V<sub>t+1</sub>.

Because this method would leave the opening positions empty for a very long time when we kick of the training, we will use a temperature variable to control how much the engine is allowed to learn from exploration moves. 


## Other differences

 - This engine uses ___ CPU and ____ GPU, while AlphaZero used 5000 TPUs
 - Everything is executed on one core, no parallelization was done
 - To speed up the learning, the terminal nodes in the MCTS return with the true value (-1, 0, 1)
