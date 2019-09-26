# MEG168-project3 Report

This project aims to build an agent to play [Frogger Game](https://en.wikipedia.org/wiki/Frogger) with reinforcement learning technique Q-Learning.   

## 1 Set Up Requirement and Command Line

Python 2.7

Run the agent by reading the trained data file in format of pickle. You can define the episode number for running this agent by --episode_number. If it is set to 1, it means that the program ends after finishing one turn of the game. 

> python Qagent.py --trained_data Nov-11_2034 --episode_number 1

If you want to train the agent, you can run this command if training from scratch. You can define the threshold of episode loss which you consider converge by --converge. 

> python Qagent.py --isTrain --converge 0.2

if you want to train the agent from a specific trained data, you can precise the data file name by --trained_data.

> python Qagent.py --isTrain --converge 0.2 --trained_data Nov-11_2034

## 2 Performance of submitted trained agent

With weights saved in the trained data file *W_Nov-11_2034.pkl*, my submitted agent could generally make **5 frogs** home. The average of the total reward the agent received per game is 10.

The agent learnt by itself a very innovative strategy to make it home safely. Rather than keeping itself in the center, the agent chose to take the most left way to get home. It is actually an very intelligent strategy because the closest row to home is moving from left to right, and starting from the left side could make sure that it can get in homes which are in the cornor. Notice that I have never given the moving direction of cars or boats to the agent, which means that it learnt this successful strategy by its exploration. I find it interesting because I never thought about this way to win the game. It shows the power of reinforcement learning: finding good solution which is unexpected by human.

<p align="center">
  <img src="https://github.com/PittCS2710/MEG168-project3/blob/master/images/performance.png" />
</p>

This trained agent is a cowardly frog. It makes effective moves only when it thinks the environment is safe enough. It needs around 10 min to complete a turn of the game.

## 3 State Space Representation

I used feature vector to represent state. 

The way to extract the feature vector X(s, a) is described as below:

- X<sub>0</sub> is 1 (constant for w<sub>0</sub>)
- If the action a is *down*, X<sub>1</sub> is the scaled distance between frog and its nearest object (car or boat or home) in the same row after taking the action a; otherwise X<sub>1</sub> is 0.  
- If the action a is *right*, X<sub>2</sub> is the scaled distance between frog and its nearest object (car or boat or home) in the same row after taking the action a; otherwise X<sub>2</sub> is 0. 
- If the action a is *up*, X<sub>3</sub> is the scaled distance between frog and its nearest object (car or boat or home) in the same row after taking the action a; otherwise X<sub>3</sub> is 0. 
- If the action a is *left*, X<sub>4</sub> is the scaled distance between frog and its nearest object (car or boat or home) in the same row after taking the action a; otherwise X<sub>4</sub> is 0. 
- If the action a is *NOOP*, X<sub>5</sub> is the scaled distance between frog and its nearest object (car or boat or home) in the same row after taking the action a; otherwise X<sub>5</sub> is 0. 
- X<sub>6</sub> is the scaled distance between frog and its nearest available home after taking the action a.
 
The reason that I extracted the feature vector in this way is that:

- Only objects in the same row as the frog could affect the frog's life. It is not necesary to condiser objects in other rows.
- Since the feature vector is related to both the observation of current environment (s) and action (a), I compute the distance after assuming a specific action a was taken.
- X<sub>6</sub> helps the frog to know how far it is from the nearest availble home. Here *available* means that there is no crocodile or frogs inside. 
- The distance must be scaled which helps Q-learning converge faster. I have scaled distance from pixel distance to a range from -1.0 to 1.0.  

Notice that the scaled distance (we call it d) used as feature is computed in a special way instead of a normal Euclidean distance. Since ideally the frog avoids cars while loving boat and home, the way to compute distance from boats or homes is exactly opposite to cars. More precisely, the distance between frog and car is considered as positive if they are not overlapped, and negative otherwise. While the distance between frog and boat or home is considered as positive if they are overlapped, and negative otherwise. I also valued how much these two are overlapped when I compute d. Since I am using linear regression (Q = WX), if w<sub>i</sub> is positive, more X<sub>i</sub>, more Q. However, if the frog is enough far away from cars, more distance doesn't actually make the frog safer (bigger Q). So I make a logical cut-off of the actual distance: if the distance is bigger than a threshold, then I set it to the maximum value 1.0. Please refer to the code for more details.
  
I have also tried some other ways to extract features:

- Setting binary value to positions in a local window around the frog. If the position is available in the current observation (s), then set it to 1, otherwise 0. This kind of feature has two problems. 1)The first one is that the actual availabllity doesn't always the same as the real situation in the next state because objects are moving in some speed. Binary feature is too hard for frogs to learn well. Features in continue value such as distance could give the frog more information. One solution is to use a bigger window which help frogs to know approaching objects. But more features bring the difficulty to train the model to converge, and they are actually not fitted to linear regression. 2)The second problem is that the local window doesn't have the information of the distance from the home. So it's hard for frog to learn moving towards homes. One solution is to include this information. I have not tried it yet but I believe that it is a good idea. The advantage of this kind of features is that it is possible to use table approach because the number of states is limited with binary representation. Even though I was using feature approach, I can easily get three frogs in home during training.

- Using unscaled distance as features. It was terrible. The agent doesn't converge at all.

- Using scaled distance but extracted in different ways from my final version. It performs less well. 

Notice that I can do nothing with the random diving turtle because there is no information about it in its observation.

In conclusion, I chose the way of extracting features which is more reasonable (easy for frog to learn) and fitted better into linear regression.

## 3 Parameterization

### 3.1 Learning rate (alpha)

I have set alpha to 0.1, 0.01 or decreasing by iteration number. 0.1 caused more oscillation than 0.01. None of them made my agent converge very well. However, the decreasing one made my agent converge a little better. But still, it doesn't converge within 600 episodes (more than 3 hours training time).

The Chapiter 18 in textbook says that a fix learning rate doesn't garantee the convergency of linear regression by Stochastic Gradient Descent while a schedule of decreasing learning rates does guarantee convergence. So I used a decreasing learning at last.

### 3.2 Exploration/exploitation trade-off

I have tested with epsilon-greedy and Upper Confidence Bounds for exploration and exploitation trade-off. 

If the epsilon is fixed to 0.1, the exploration processes quite slow. Especially when I set the inital W to zeros, the agent doesn't even move. In addition, I have tried a decreasing epsilon from 1 to small value during the training process. However, it doesn't perform well because the agent could block in the beginning line after a huge number of training iterations if there is a very little probability for exploration. 

Inspired by multi-armed bandit problem that I learnt before, I haved tried the Upper Confidence Bounds (UCB) algorithms shown on this [slide](https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf). I chose the action that maximises Q(s,a) + sqrt(2\*ln(N)/N<sub>a</sub>). In this frog game, I define N is the total number of times visiting this frog position, the N<sub>a</sub> is the total number of choosing action a in this frog position. With UCB, the frog explores very much at the begining of the training and it made quite a lot exploration even after a lot of iterations because that the log(N) increases as well during training. My agent learnt effectively with UCB algorithm.

### 3.3 Reward discount constant (gamma)

I have used 0.9 and didn't try other values.

## 4 Convergence Analysis

I trained the agent from scratch for more than 3 hours. The agent run more than 600 episodes. One episode means from the begining of one turn of the game to the end (die or get all frogs in home). I define loss function of each step as (Q<sub>sample</sub> - Q<sub>old</sub>)<sup>2</sup>/2. The episode loss is defined by the sum of loss of all steps in this episode. The image below shows the episode loss in each episode. It keeps oscillating and doesn't converge. However, 5 frogs begun getting into homes only after around 50 episodes of training. I considered it as "converge" if the loss episode is less than 0.2 (the line in orange). Then it "converged" at episode 158 for the first time and gave us the trained data file *W_Nov-11_2034.pkl* which could consistently send 5 frogs home. 
<p align="center">
  <img src="https://github.com/PittCS2710/MEG168-project3/blob/master/images/loss.png" />
</p>

I also plot weights from w<sub>0</sub> to w<sub>6</sub> in each episode and we can see that they don't have a trend to converge.

![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_0.png)
![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_1.png)
![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_2.png)
![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_3.png)
![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_4.png)
![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_5.png)
![image](https://github.com/PittCS2710/MEG168-project3/blob/master/images/W_6.png)


## 5 Conclusion

The agent learnt effectively with the features that I extracted and the trade-off strategy. I didn't change any reward, but it could begin send frog home only after several episodes and successed to send 5 frogs home after less than 20 minutes of training. The agent did a good job with Q-learning and learnt an innovative strategy to get into homes.

The problem is that the agent doesn't converge to a minimum loss and always oscillate. 

## 6 Perspectives

Don't include the information that the frog should *avoid* cars and *gets in* boats or homes in the feature vector. The frog should learn it by itself. It would be more general if I use three features (scaled distance from nearest cars in the same row, scaled distance from nearest boat in the same row and scaled distance from nearest home in the same row) instead of one feature (scaled distance from the nearest object in the same row). 

Try more gamma values for exprimentations.

Train more time to see if it will eventually converge or not. Study how to make it converge.

Test table based approach.

Try Deep Q-learning.

## Take-away points from this project

I have learnt some training tricks while doing this porject:

- A good trade-off policy could help training the agent much faster. UCB is an efficient one.

- We need to scale well the feature. Reasonable features make learning process faster.

- I have tried the techinique of Experience Replay presented in this [slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf) but it didn't help in this game. I find a problem with Experience Replay: previously generated states and rewards are selected more often for updating weights than recently generated ones if we select samples with a uniform probability distribution. According to Bellman Equation, the optimal Q(s,a) is the estimation of Q<sub>sample</sub>. If we change the distribution of Q<sub>sample</sub>, then the sampling process no more respect the accurate estimation. 


