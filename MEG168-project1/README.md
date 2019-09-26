# MEG168-project1 Report 

This project aims to solve three puzzles (Jugs, Path-Planning, Burnt Pancakes) with several search algorithms, including brute-force search strategies and heuristic search strategies. 

## 1 Set up requirement

The experimentation is done on MacBook Pro 2.7 GHz quad-core Intel Core i7, 16 GB 2133 MHz LPDDR3.

Python version 3.6.5

numpy==1.14.3

dataclasses==0.6

typing==3.6.4

## 2 References

[1] Bulteau L, Fertin G, Rusu I. Pancake flipping is hard[J]. Journal of Computer and System Sciences, 2015, 81(8): 1556-1574.


## 3 Implementation

### 3.1 Water Jugs Problem

#### 3.1.1 Successor states generating order

Respect the jugs' order as its input config file. 

For each jug i:

  - Empty jug i if it is not already empty.  
  - Fill jug i if it is not already full.
  - Transfer jug i to each of other jugs j (j != i) if jug i is not empty and jug j is not full.
  
Take test_jugs.config as example where the capacities of two jugs are (4, 11):

The generating order of successor states from the state (2, 5) is:
(0, 5), (4, 5), (0, 7), (2, 0), (2, 11), (4, 3).

The generating order of successor states from the state (4, 0) is:
(0, 0), (0, 4), (4, 11).

My implementation of this puzzle is not only adaptable for 2 or 3 jugs, but also for more than 3 jugs.

#### 3.1.2 Heuristic function

My heuristic function named "sum" firstly computes the total gallons of water in jugs in the current state. The estimated distance to the goal state is equal to the difference between the sum of water in current state and goal state.

Take test_jugs.config as example where the goal state is (1, 0): 
The heuristic score of state (2, 5) is equal to |(2 + 5) - (1 + 0)| = 6.

The heuristic function is generally reasonable because smaller the score, closer to the goal. 

- Admissible? No.

The heuristic score of state (1, 11) is equal to 11. However, it takes only one step (path cost = 1) to achieve the goal state by emptying the second jug.

- Consistent? No.

If a heuristic function is not admissible, then it is not consistent neither.

#### 3.1.3 Search Strategy Discussion

Since the proposed heuristic function is neither admissible nor consistent and the brute force search space of this problem is reasonable, I prefer choosing uninformed search algorithm as the best strategy. So I compare these four factors between BFS, DFS and IDDFS. I don't consider UniCost because the step cost for this problem is always equal to 1.

Notice that the DFS implemented in my project is complete for this jugs problem because it checks new states against those on the path from the root to the current node, which avoids infinite loops in finite state spaces.

The table below evaluates search strategies. b1 is the branching factor of graph_search, b2 is the branching factor of tree_search; d is the depth of the shallowest solution; m is the maximum depth of the search tree. 

|                  | BFS | DFS | IDDFS |
|------------------|-----|-----|-------|
| Completeness     | Yes | Yes | Yes   |
| Optimality       | Yes | No  | Yes   |
| Time complexity  | O(b1^d) | O(b2^m) | O(b2^d)   |
| Space complexity | O(b1^d) | O(b2*m)  | O(b2*d)    |

I will not choose DFS because it is not optimal. For the 2-jugs problem, b1 (around 1.5) is smaller than b2 (around 2), so BFS's time complexity is better than IDDFS. As for space complexity, IDDFS is much better than BFS because it doesn't need to store explored list or all the generated nodes. Experimental results are coherent with the complexity analysis. In conclusion, BFS and IDDFS are both good choices for this problem. However, if we meet the space limit rather than time limit (usually the case), IDDFS is better.

### 3.2 Path-Planning Problem

#### 3.2.1 Successor states generating order

The successor state of the current state is generated according to the order of possible paths in the input config file. If the possible path contains the current state, then I generate its destination/depart city as successor state.

Take test_cities.config as example:

The successor states of C00 is in order of: (C10, 7), (C01, 5), (C11, 4) 

The successor states of C01 is in order of: (C11, 7), (C00, 5), (C02, 5), (C12, 4), (C10, 2)

#### 3.2.2 Heuristic function

I used the straight-line distance (Euclidean distance) heuristic function named by "sld".

#### 3.2.3 Search Strategy Discussion

Since the step costs of this problem are not equal, BFS, DFS and IDDFS is not adaptable for this problem. So I compare these four factors between UniCost, Greedy, ASTAR and IDASTAR. The table below evaluates search strategies. I use experimental results for evaluating time and space complexity.

Since the straight-line distance heuristic function is consistent (and surely admissible), graph-based A* is optimal. The Greedy search is complete here because my implementation is using graph-search and the search spaces are finite. 

|                  | UniCost                    | Greedy                  | A*                         | IDA*        |
|------------------|----------------------------|-------------------------|----------------------------|-------------|
| Completeness     | Yes                        | Yes                     | Yes                        | Yes         |
| Optimality       | Yes                        | No                      | Yes                        | Yes         |
| Time complexity  | 261                        | 24                      | 230                        | 23186       |
| Space complexity | frontier 121, explored 142 | frontier 21, explored 4 | frontier 114, explored 120 | frontier 28 |

Even though Greedy has less time and space complexity, it is not a good choice because it not optimal. A* is better than UniCost regarding both time and space complexity with a consistent heuristic function. IDA* needs much less space but it demands more time complexity. In conclusion, A* and IDA* are both good choices for this problem. However, if we meet the space limit rather than time limit (usually the case), IDA* is better. 

### 3.3 Burnt Pancake Problem

#### 3.3.1 Successor states generating order

Insert the spatula from closest to the bottom to closest to the top.

Take test_pancakes1.config as example:

The order of successor states of [-1, -11, -3, -6, -9, -4, -7, -10, -5, -8, -2] is:

[(2, 8, 5, 10, 7, 4, 9, 6, 3, 11, 1), (8, 5, 10, 7, 4, 9, 6, 3, 11, 1, -2), (5, 10, 7, 4, 9, 6, 3, 11, 1, -8, -2), (10, 7, 4, 9, 6, 3, 11, 1, -5, -8, -2), (7, 4, 9, 6, 3, 11, 1, -10, -5, -8, -2), (4, 9, 6, 3, 11, 1, -7, -10, -5, -8, -2), (9, 6, 3, 11, 1, -4, -7, -10, -5, -8, -2), (6, 3, 11, 1, -9, -4, -7, -10, -5, -8, -2), (3, 11, 1, -6, -9, -4, -7, -10, -5, -8, -2), (11, 1, -3, -6, -9, -4, -7, -10, -5, -8, -2), (1, -11, -3, -6, -9, -4, -7, -10, -5, -8, -2)]

#### 3.3.2 Heuristic function

This paper[1] mentions the concept of breakpoints: Consider a sequence S of length n, S = <􏰃x<sub>1</sub>, x<sub>2</sub>, . . . , x<sub>n</sub>􏰄>. Sequence S has a breakpoint at position r, 1≤r<n if x<sub>r</sub>!=x<sub>r+1</sub>-1 and x<sub>r</sub>!=x<sub>r+1</sub>+1. It has a breakpoint at position n if x<sub>n</sub>!= n.

I think it could be a heuristic function for this problem and named it as "bp". 

The heuristic function of the example state is equal to 11. 

- Admissible? Yes.

We need to insert at lease one spatula to decrease a breakpoint. It never overestimates the cost to reach the goal.

- Consistent? Yes.

Since each step always costs 1 and each step can decrease at most 1 breakpoint, for every node n and every successor n′ of n, we have h(n) ≤c(n, a, n′) + h(n′). 

According to my experimental results of the pancakes test 1, the solution by IDASTAR and ASTAR are same. We know that the IDASTAR search is complete and optimal, and graph-based ASTAR is optimal if and only if its heuristic function is admissible and consistent, so my experimental results confirm that the breakpoint heuristic function is admissible and consistent.

#### 3.3.3 Search Strategy Discussion

Since step costs are equal, I will not consider UniCost. The search spaces for this problem are finite. My implementations of DFS and Greedy are complete for this problem, as what I presented above for other problems. 

The table below evaluates search strategies for the pancakes test 1 configuration file. For this problem the branching factor of tree-search is 11 because each state can have 11 successor states. Since the state space has 11! possibilities (which is a huge number), we suppose that duplicate states rarely appear, so the branching factor of graph-search is also considered as 11. Experimental result of A* or IDA* shows that the depth of the shallowest solution is 15. m is the maximum depth of the search tree, which could be a really large number (15<m<11!).  

|                  | BFS   | DFS  | IDDFS | Greedy                     | A*                              | IDA*        |
|------------------|-------|------|-------|----------------------------|---------------------------------|-------------|
| Completeness     | Yes   | Yes  | Yes   | Yes                        | Yes                             | Yes         |
| Optimality       | Yes   | No   | Yes   | No                         | Yes                             | Yes         |
| Time complexity  | O(11^15) | O(11^m) | O(11^15) | 797                   | 889870                          | 1059045     |
|Space complexity| O(11^15)| O(11*m) | O(11*15) | Frontier 718,  explored 80 | Frontier 797457, explored 92414 | Frontier 91 |

As the table shows, the time complexity of BFS is in order of 10^15. If 10^5 nodes could be searched per second, the running time of BFS should be 10^4 days, which is not reasonable. So BFS and IDDFS are not good search algorithms for this problem. Even though its space complexity is better than other brute force search, DFS is not in consideration because its time complexity is even worse and it is not optimal. The best search algorithms should use heuristic functions to reduce time and space complexity. My experiments with A* and IDA* are under 7 seconds, so the time complexity is really reasonable. Greedy is faster but it is not optimal. Greedy could be a good choice when the length of pancakes test becomes big, such as the test 2 configuration file. IDA* 's time complexity is comparable with A*, however, its space complexity is much smaller. 

In conclusion, if the number of pancakes is not big, such as the test 1, then IDA* is the best search algorithm; if the number of pancakes is big, such as the test 2, then Greedy is a good choice even though its solution is not optimal.

Moreover, I would like to estimate the running time of test 2 by A* and IDA*. According to the result of test 1, the branch factor with heuristic function could be estimated as e^(ln889870/15) which is equal to 2.5. Therefore, 11^_epsilon_ = 2.5, then _epsilon_ is equal to 0.4. The length of pancakes in test 2 is 40, so the time complexity of test 2 is 40^0.4^d, where d is the shallowest depth of goal state. Assuming d is also 15, the time complexity of test 2 is in order of 10^9. The running time of test 2 should be 1000 times more than test 1. As a result, the estimated running time of test 2 is more than 1 hour. It explains why my experimentation of test 2 by A* or IDA* can not finish in 30 minutes. 

### 3.4 Search Algorithms Implementation

- BFS: 

  Graph-search based Breadth First Search. Append child node to frontier list only if the state of the child node is neither in the explored list nor same as nodes' state in the frontier list.
  
- DFS:

  Tree-search based Depth First Search. Append child node to frontier list only if the state of the child node is different from all of its parent nodes' states.
  
  Notice that I used data structure _queue_ to implement the frontier list of all the deep-based search algorithm. For example, if the node A generates B, C, D children nodes in order, then the next node to be checked and removed from frontier list is D. I also implemented in the same way for IDDFS and IDASTAR. I will not repeat it in the following.
  
- IDDFS:
  
  Tree-search based Iterative Deepening Depth-first Search. No explored list. Append child node to frontier list only if the state of the child node is different from all of its parent nodes' states.
  
- UniCost:

  Graph-search based Uniform-Cost Search. Append child node to frontier list only if the state of the child node is not in the explored list. I use priority queue for the frontier list. Even though a child node's state is same as ones in the frontier list, the node with the smallest path cost will be pop out at first.
  
- Greedy:

  Graph-search based Greedy Search. Similar to UniCost except for using heuristic function score as the priority.
  
- ASTAR:

  Graph-search based A* Search. Similar to Greedy except for using sum of heuristic function score and path-cost as the priority.

- IDASTAR:

  Tree-search based Iterative-Deepening A* Search. No explored list. Append child node to frontier list only if the state of the child node is different from all of its parent nodes' states.
  
## 4 Experimentation:

More detail is in the file named by "typescript".

#### Run my puzzle solver for water jugs

* BFS:

  > python puzzlesolver.py config/test_jugs.config bfs

  Outputfile is named by: output/jugs_by_bfs.output

* DFS:

  > python puzzlesolver.py config/test_jugs.config dfs
  
  Outputfile is named by: output/jugs_by_dfs.output
  
* Greedy with heuristic function named "sum":

  > python puzzlesolver.py config/test_jugs.config greedy --heuristic_function sum
  
  Outputfile is named by: output/jugs_by_greedy_with_sum.output
  
#### Run my puzzle solver for cities path problem

* Unicost:

  > python puzzlesolver.py config/test_cities.config unicost
  
  Outputfile is named by: output/cities_by_unicost.output
  
* Greedy with heuristic function named "sld":

  > python puzzlesolver.py config/test_cities.config greedy --heuristic_function sld
  
  Outputfile is named by: output/cities_by_greedy_with_sld.output
  
* Astar with heuristic function named "sld":

  > python puzzlesolver.py config/test_cities.config astar --heuristic_function sld
  
  Outputfile is named by: output/cities_by_astar_with_sld.output
  
#### Run my puzzle solver for pancakes problem

* IDDFS:

  > python puzzlesolver.py config/test_pancakes1.config iddfs
  >
  > python puzzlesolver.py config/test_pancakes2.config iddfs
  
  The search strategy takes longer than 30 minutes. I interrupted it and because the search strategy was not able to finish within the allotted time. 
  
* Astar with heuristic function named "bp":
  
  > python puzzlesolver.py config/test_pancakes1.config astar --heuristic_function bp
  >
  > python puzzlesolver.py config/test_pancakes2.config astar --heuristic_function bp
  
  Outputfile for pancakes1 is named by: output/pancakes1_by_astar_with_bp.output
  
  The search for pancakes2 takes longer than 30 minutes. I interrupted it and because the search strategy was not able to finish within the allotted time.
  
* IDAstar with heutistic function named "bp":

  > python puzzlesolver.py config/test_pancakes1.config idastar --heuristic_function bp
  >
  > python puzzlesolver.py config/test_pancakes2.config idastar --heuristic_function bp
  
   Outputfile is named by: output/pancakes1_by_idastar_with_bp.output
   
   The search for pancakes2 takes longer than 30 minutes. I interrupted it and because the search strategy was not able to finish within the allotted time.
  
## 5 Communication:

- Asked some classmates if they can run pancakes2 under 30 minutes. (I can't remember all of thier names)
- Helped some classmates debugging by chat without seeing their codes. They eventually found out thier bug by themselves.
- Checked with Maher if the optimal step cost of pancakes1 is equal to 15.
- Yaohan checked with me if the optimal step cost of cities is equal to 22.
