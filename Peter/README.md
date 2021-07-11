# DecentralizedCoverage
Implementation of Decentralized, Adaptive Coverage Control for Networked Robots



<u>**Experiments Using Consensus**</u>

Experiment 1: Tests whether or not Agent parameters can converge to optimal when using parameter consensus with weighting according to Remark 5 

- Hyperparameter Overview: 
  - 6 agents, 2 gaussian basis functions, consensus parameters with weighting of 1, checks positive definiteness on each iteration 
- Results: 
  - Agents converge to estimated centroids and converge to true centroids 
  - Agent sensing parameters converge to true sensing parameters 
  - Gaussian basis functions are not positive definite
  - Results are consistent with Corollary 2 



Experiment 2: Tests whether or not Agent parameters can converge to optimal when using parameter consensus that weights parameter differences by distance to voronoi neighbor 

- Hyperparameter Overview: 
  - 6 agents, 2 gaussian basis functions, consensus parameters weighted by dist to voronoi neighbor, checks positive definiteness on each iteration 
  - Significantly lowered positive consensus gain (zeta) relative to its value in experiment 1
- Results: 
  - Agents converge to estimated centroids and converge to true centroids 
  - Agent sensing parameters converge to true sensing parameters  
  - Gaussian basis functions are not positive definite 
  - Results are consistent with Corollary 2



Experiment 3: Tests the effects of lowering the variance of the gaussian basis functions 

- Hyperparameter Overview:
  - Same exact hyperparameters as Experiment 2, but variance set to 1.0 instead of 10.8 and the number of iterations is set to 50 instead of 300
- Results:
  - Agents converge to estimated centroids and converge to true centroids 
  - Agent parameters do not converge to the true sensing parameters 
    - They decrease to a minimum, then blow up towards infinity 
  - Gaussian Basis functions are not positive definite 
  - Results are consistent with Corollary 2
  - I didn't record the experiment, but agent parameters converge with variance set to 5.0



Experiment 4: Tests the effects of increasing consensus gain 

- Hyperparameter Overview:
  - Same exact hyperparameters as Experiment 2, but positive consensus gain (zeta) set to 0.1 instead of 0.005 and the number of iterations is set to 50 instead of 300
- Results:
  - Agents do not converge to their estimated or true centroids 
  - Agent parameters do not converge to the true sensing parameters 
    - They decrease to a minimum, then blow up towards infinity 
  - Gaussian basis functions are positive definite 
    - Increasing positive consensus gain (zeta) can overwhelm the first two terms of (20). Even if the conditions for Corollary 2 are met, a bad positive consensus gain value can prevent the agents' parameters and positions from converging 



<u>**Experiments Not Using Consensus**</u>

Experiment 1: 

