# Reinforcement Learning Submission

## Question 1: Value Iteration

Implemented batch value iteration in `ValueIterationAgent`. Each iteration computes a new value table from the values of the previous iteration, using the Bellman update over all legal actions and transition probabilities. Terminal states remain at value `0`.

## Question 2: Bridge Crossing Analysis

Updated `question2` in `analysis.py` to make the agent prefer crossing the bridge. The discount remains `0.9`, and the transition noise is reduced to `0.0` so the agent is willing to take the risky bridge path.

## Question 3: Discount Grid Analysis

Filled in all five parameter settings in `analysis.py`:

- `question3a`: prefers the close exit while risking the cliff.
- `question3b`: prefers the close exit while avoiding the cliff.
- `question3c`: prefers the distant exit while risking the cliff.
- `question3d`: prefers the distant exit while avoiding the cliff.
- `question3e`: avoids terminal exits by using a positive living reward.

## Question 4: Q-Value Computation From Values

Implemented `computeQValueFromValues` so each Q-value is computed as the expected reward plus discounted future value across all possible next states.

## Question 5: Action Selection From Values

Implemented `computeActionFromValues` so the value iteration agent chooses the action with the highest computed Q-value and returns `None` for terminal states with no legal actions.

## Question 6: Q-Learning

Implemented the tabular `QLearningAgent` using a `util.Counter` for Q-values. The agent now supports default zero Q-values, max-value lookup, best-action selection with random tie-breaking, epsilon-greedy exploration, and the standard Q-learning update rule.

## Question 7: Epsilon-Greedy Action Selection

Implemented epsilon-greedy behavior in `getAction`. The agent chooses a random legal action with probability `epsilon`, otherwise it follows the learned policy. If no legal actions exist, it returns `None`.

## Question 8: Bridge Crossing With Q-Learning

Returned `NOT POSSIBLE` because the requested behavior cannot be guaranteed by only tuning epsilon and learning rate under the assignment constraints.

## Question 9: Pacman Q-Learning

The completed `QLearningAgent` supports `PacmanQAgent`, allowing Pacman to learn from training games and then act greedily during testing after exploration and learning are disabled.

## Question 10: Approximate Q-Learning

Implemented `ApproximateQAgent` so Q-values are computed as the dot product of feature values and learned weights. The weight update uses the temporal-difference error multiplied by each feature value and the learning rate.

