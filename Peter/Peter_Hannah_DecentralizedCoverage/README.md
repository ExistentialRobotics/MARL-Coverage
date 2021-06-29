# DecentralizedCoverage
Implementation of Decentralized, Adaptive Coverage Control for Networked Robots



Experiment 1: Tests whether or not Agent parameters can converge to optimal when using parameter consensus 

- Hyperparameter Overview: 
  - 6 agents, 2 gaussian basis functions, consensus parameters with weighting of 1, checks positive definiteness on each iteration 
- Results: 
  - Agents converge to estimated centroids and converge to true centroids 
  - Agent sensing parameters converge to true sensing parameters 
  - Gaussian basis functions are positive definite on each iterations 
  - Results are consistent with Corollary 2 



Experiment 2: Tests whether or not Agent parameters can converge to optimal when using parameter consensus that weights parameter differences by distance to voronoi neighbor 

- Hyperparameter Overview: 
  - 6 agents, 2 gaussian basis functions, consensus parameters weighted by dist to voronoi neighbor, checks positive definiteness on each iteration 
- Results: 
  - Agents converge to estimated centroids and converge to true centroids 
  - Agent sensing parameters converge to true sensing parameters  
  - Gaussian basis functions are positive definite on each iterations 
  - Results are consistent with Corollary 2
