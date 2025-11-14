
## General Machine Learning Concepts
### Bias Variance Trade-off
#### Regularization
### Common Distributions
#### Discrete
* Bernouilli 
* Binomial
* Multinouilli 
* Multinomial
* Poisson
* Empirical

#### Continuous
* Gaussian
* Student t
* Laplace 
* Gamma
* Beta
* Pareto
* Dirichlet
### Curse of Dimensionality
----
### Decision Theory
### Ensemble Learning
#### Boosting
#### Bootstrapped Aggregation (Bagging)
#### Stacked Generalization (Blending)
#### Averaging Generalization 
### Estimators
#### Maximum Likelihood Estimation
#### Maximum A Posteriori Estimation (MAP)
#### Bayesian Modeling
### Evaluation Metrics
#### Classification Metrics
----
#### Regression Metrics
### Frequentist vs Bayesian
* :mag: <span class='note'> Side Notes </span> :

    * Don't get mistaken: using Bayes rule doesn't make you a Bayesian. As my previous ML professor [Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/) used to say: "If you're not integrating, you're not a Bayesian". :sweat_smile:

    * If you understood well the point of view of frequentist, you might be surprised of seeing something like $p(x\| \Theta)$, which means the "conditional distribution of *x* given $\Theta$". Indeed for frequentists $\Theta$ is not a random variable and thus conditioning on it makes no sense (there's a single value for $\Theta$ which may be unknown but is still fixed: it's value is thus not a condition). Frequentists would thus write such distributions: $p(x;\Theta)$ which means "the distribution of *x* parameterized by $\Theta$". In statistics and machine learning, most people use $\|$ for both cases. Mathematicians tend to differentiate between the notations. In this blog, I will use $\|$ for both cases in order to keep the same notation as other ML resources you will find. 

### Generative vs Discriminative 
-----
### Information Theory
-----
### Model Selection
#### Cross Validation
#### Hyperparameter Optimization
### Monte Carlo Estimation
### No Free Lunch Theorem
----
### Parametric vs Non Parametric
-----
### Quick Definitions
**Capacity**
**Convex functions**
**Exponential Family**
**Inference**
**Kernels**
**Norms**
**Online learning**
**Surrogate Loss Function**








## Supervised Learning
### Classification
#### Decision Trees
----
#### Logistic Regression 
<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Probabilistic</a>
    </div>
  </div>
</div>
#### Softmax 
<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Probabilistic</a>
    </div>
  </div>
</div>
#### Support Vector Machines (SVM)
<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Non-Probabilistic</a>
    </div>
  </div>
</div>

* :mag: <span class='note'> Side Notes </span> :
    * There are extensions such as [Platt scaling](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639) to interpret SVM in a probabilistic manner.

#### Gaussian Process
  
----
Now the Generative classifiers

----

#### Gaussian Discriminant Analysis
#### Gaussian Mixture Model
#### Latent Dirichlet Allocation 


### Regression
#### Linear Models
##### Linear Regression
Normally use OLS but could use something else to estimate
##### Ordinary Least Squares
estimation technique: often used in Linear regression
#### Decision Trees
----
#### Gaussian Processes

### Ranking
## Unsupervised Learning

### Clustering
Nota Bene: Careful when ensemble learning to label switching
#### Density Based Clustering
#### Hierarchical Clustering
#### K-Means
#### K-Nearest Neighbors (KNN)
#### Spectral Clustering

### Density Estimation
#### Collaborative Filtering
Recommender systems (also content based or hybrid)

### Dimensionality Reduction
#### Autoencoders
#### Independent Component Analysis (ICA)
#### ISOMAP
#### Linear Discriminant Analysis (LDA)
#### Multidimensional Scaling (MDS)
#### Principal Component Analysis (PCA)
#### Projection Pursuit
#### Sammon Mapping
#### Self-Organizing Maps
#### T-SNE


### Outlier Detection
Nota Bene: distinguish global and local outliers
##### Cluster-Based
##### Distance-Based
##### Graphical Approaches
##### Model-Based


## Reinforcement Learning
### Exploration vs Exploitation
---
### Markov Decision Process
---
### Dynamic Programming
---
### Monte Carlo Methods
---
### Temporal Difference Learning
---
#### Expected Sarsa
#### N steps Expected Sarsa
### Eligibility Traces TD($\lambda$)
### Policy Gradient Methods
### Model Based RL
### Function Approximation
#### Deep Reinforcement Learning
### Links to Neuroscience


## Partially supervised learning
non t called like that
### Active Learning
### Semi-supervised learning


## Approximate Inference
### Deterministic
#### Variational Inference
Comes from Calculus of variation
- standard calculus finds derivative of functions (mapping of input values of a variable to output value). Derivative gives inenitiisimal change in output due to inifnicsimal input change. 
- functional takes function as input. For example entropy takes a probability distribution as input. so functional derivative is infinitisimal change in output due to change in pinput function. basically same rule applies

variational inference is simply max/min in function space, and restrict your input space to some family
##### Mean Field

Use only factorized distribuion 

##### Normalizing Flows

Take a simple distribution and then transform the samples to make it mroe coplicated : 

https://akosiorek.github.io/ml/2018/04/03/norm_flows.html

### Stochastic
Sampling Based
#### MCMC


## Deep Learning
### Backpropagation
### Feed-forward Neural Networks
### Convolution Neural Network
### Recurrent Neural Network
### Autoencoders
### Regularization
* Early Stopping
* Dropout
* Multi-task Learning
* Norm Penalties

### Deep Generative Models
#### Variational Autoencoders
-----
#### Generative Adversarial Networks
#### Neural Autoregressive 
#### Flow-based deep generative models


## Graphical Models
### Directed Graphical Models (Bayes Networks)
### Undirected Graphical Models (Markov Networks)
#### Pair Wise Undirected Graphical Models
#### Gibbs Networks
#### Conditional Random FIelds
- https://prateekvjoshi.com/2013/02/23/what-are-conditional-random-fields/
- generalization of multinomial listic regression to seuence prediction (i.e y's are dependent)
- https://www.youtube.com/watch?v=GF3iSJkgPbA
- http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/
- discriminative probabilisticc
- advantage over HMM for classification because learn p(y|x) instead of joint
- use for labeling sequences
- using conditional probabilities is udes to relax the conditional independence of naive bayes (like logistic regression) or the conditional indep required in HMM
- HMM are subcase of CRF  where constant probabilities are used to model state transitions. 


## Natural Language Processing


## Time Series


## Optimization



### Convex Optimization
#### Linear Programming
##### Canonical and Slack form
----
##### Duality
##### Simplex Algorithm

### Non Convex Optimization

#### Bayesian Optimization
#### Robust Optmization
#### Evolutionary Methods
Compare to RL : If the space of policies is sufficiently small, or can be structured so that good policies are common or easy to find or if a lot of time is available for the search then evolutionary methods can be effective. In addition, evolutionary methods have advantages on problems in which the learning agent cannot sense the complete state of its environment.
##### Genetic algorithms
#### Simulated annealing
#### Tabu Search
#### Hill climbing with random restart
#### Nelder-Mead heuristic

## Information Theory
### Kolmogorov Complexity
the Kolmogorov complexity of an object, such as a piece of text, is the length of the shortest computer program  that produces the object as output

### Minimum Description Length
Suppose we consider a number of different hypotheses for some phenomenon.
We have gathered some data that we want to use somehow to evaluate these
hypotheses and decide which is the“best”one. The minimum description length (MDL) criteria in machine learning says that the best description of the data is given by the model which compresses it the best. Put another way, learning a model for the data or predicting it is about capturing the regularities in the data and any regularity in the data can be used to compress it. Thus, the more we can compress a data, the more we have learnt about it and the better we can predict it. n complexity and goodness of fit

MDL is also connected to Occam’s Razor ( principle of parsimony) used in machine learning which states that “other things being equal, a simpler explanation is better than a more complex one.” In MDL, the simplicity (or rather complexity) of a model is interpreted as the length of the code obtained when that model is used to compress the data. The ideal version of MDL is given by the Kolmogorov Complexity.

L_{MDL}(D) := min_{H \in \mathcal{H}} (L(H) + L(D|H))

Bayesian inerpretation: looking to minimize −[log prior + log likelihood]. => same as looking for MAP

### Information Bottlneck

the idea is to compress while keeping all the "important" infrormation

by maximizing the mutual information of X and its compressed counterpart Z (they call x tilda), you would keep the more infromation you can in a unsupervised way. But you can do in a supervised may by maximizing I(Z,Y) where Y is a sueprvised task (e.g. classification). If you wanted to maximize this you would simply not compress at all and set Z = X. But you want to compress maximize under the constraint that I(Z,X) < gamma. Using lagrange multipliers, this turns into max(I(Z;Y) - beta I(Z;X)). => effectively also maximizing the compression. => maximizing bottleneck by forcing Z and X to be different distributions.  Note that cannot just minimize entropy of Z because for some tasks a very small entropy is enough to encode all. beta gives the tradeoff between accuracy and compresssion 

Note that often actually minimize  - 1/beta * (I(Z;Y) - beta I(Z;X) ) =  I(Z;X) - 1/beta I(Z;Y)  =  I(Z;X) - beta' I(Z;Y)  which is equivalent

an other simpel way of seeing it, is to maximize compression while keeping all the important information:

min I(Z;X)
s.t. H(Y|X) = H(Y|Z) 

R/ compression can either be in size (ex: autoencoder) but also in information Ex: dropout / noisy atoencoder => not less but add noise

Note that to do this, it is assuemd that you have p(x,y) which is a hard task on its own (i.e ML)

link with rate distortion theory: 
if want to compress input to max R I(z;x), what is the lowest distortion  D that you can get =>
E_{x,z}[D(x,z)] s.t. I(Z;X) <= R. 
if D(x,z) = KL(p(y|x) || p(y|z)), then rate distortion is equivalent to information bottleneck

### FIsher Information

If we have a known density p(X|theta), then information asnwers the question "how useful is X to determine theta". i.e. amount of information that an observable random variable X carries about an unknown parameter θ.

If f is sharply peaked with respect to changes in θ, then there's only a small range of possible theta that could explain data => doesn't need much data. => variance of parameters in sharp distribution is smaller than flat. This suggests studying some kind of variance with respect to θ.

If very shap then curve is high => high hessian / second derivative => low variance (donc proportional to inverse of hessian). Fisher information is hessian of log likelihood. hus, the Fisher information may be seen as the curvature of the support curve (the graph of the log-likelihood). Near the maximum likelihood estimate (because look at derivative at MLE), high Fisher information indicates that the maximum is sharp.
=> can be used to compute confidence intervals in frequentist statistis

Formally, the partial derivative with respect to θ of the logarithm of the likelihood function p(X|theta) is called the “score”.  if θ is the true parameter, it can be shown that expectation is 0. Fisher information is the variance => expectation of squared derivative of log likelohood. FOr high dimension hessian.

#### Cramer Rao Bound
The Cramér–Rao bound states that the inverse of the Fisher information is a lower bound on the variance of any unbiased estimator of theta
var(\hat theta) = I^{-1}(theta)

Note that this is similar to : "What’s the distribution of the unknown parameter?". but with frequentist view => makes no sense to talk about distribution of parameters.

Source: 
- https://www.quora.com/What-is-an-intuitive-explanation-of-Fisher-information
- https://www.youtube.com/watch?v=i0JiSddCXMM


## Computational Neuroscience
### Spiking Neural Networks

## Other
### Causal Learning
#### Counter Factuals
### State Space Models
### Optimal control 
https://vincentherrmann.github.io/blog/wasserstein/
we look at the minimal amount of energy to move one distribution to an other (it's similar to KL divergence but instead of looking at the difference in the y axis (i.e difference of probability given to each point in both distribution) you look at the distance in x axis (i.e how much energy does it take to make one one distribution turn itno an other). For example if there are 2 very non linear pdf that are the same but shifted by a small amout, the KL might be very big (because difference of probability given to each point might be big) but not the earth moving disatnce) 
### Useful Tricks
#### Density Ratio Estimation
Many tasks in probabilistic ML, require estimation of ratio of probabilities:
* importance sampling: p(x)/q(x)
* KL divergence E_p[log(p(x)/q(x))]
* COnditional probability: p(y|x)=p(y,x)/p(x)
* hypothesisi testing : r(x)=p(x|H_0)/p(x|H_1)

often the probabilities are unkown (or in   tractable), so can't compute numerator and denom eparately then divide, we may only have samples drawn from the two distributions, not their analytical forms. Dnsity ratio estimation says that can sompute the ratio using a dbinary discriminator D that says if x is from p:
r(x)=p(x)/q(x)= D(x)/(1-D(x))
=> if have samples can simply train discriminator and woudl get the ratio.

info:
- intuitive explanantion: http://blog.shakirm.com/2018/01/machine-learning-trick-of-the-day-7-density-ratio-trick/
- more details : http://yosinski.com/mlss12/media/slides/MLSS-2012-Sugiyama-Density-Ratio-Estimation-in-Machine-Learning.pdf
- code and plots : http://louistiao.me/notes/a-simple-illustration-of-density-ratio-estimation-and-kl-divergence-estimation-by-probabilistic-classification/


## Probabilities 
### Definition
### Gaussian
### Bernoulli
### Exponential
### Poisson
### Exponential Family
### Distribution 
### Processes
2 explanations 

* Random process is a sequence of random variables. To answer any question we might be interesting in, we need to know the mean and variance of every variable, but this is not all. Indeed, r.v. can be dependent so we need to know the joint distribution of every subset of r.v. e.g. P(X_1, X_3, X_9)?
* Treat the whole èrpcess as a single r.v. where the output is any possible sequence/path/function (e.g. 0 1 for bernoulli / randov variable over function for gaussian process). With this view can start asking questions about infinit sequence. E.g. in bernoulii : proba of all 1? Would be 0 if p in [0,1[. Furthermore every sequence has proba -> 0 (because multiply proba that are less than 1), but there's an infinite number of possible => sum of all is still 1. It's like when continuous r.v.
 
### Bernoulli Process
Collection of i.i.d bernoulli r.v. => joint of any subset is simply product

* discrete time
* number of arrivals in n time slots : binomial pmf
* interarival time : geometric pmf
* time to k interval : pascal pmf
* memoryless (once arrival comes, proba to next arrivl is indep. I.e. as if process start again at every time step)

info : https://www.youtube.com/watch?v=gMTiAeE0NCw&index=13&list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8

### Poisson Process

* geenralize bernoulli process to continuous => if counting number of clientd that came, instead of putting a 1 or 0 at every second you want to be more precise (what if 2 client come in same second) => go to milisieconb => go to continuous => no more slots, simly write down time when client came
* time homogeneitynumber of clients per interval only depend on time. But every "time slot" (although continuous) has same proba of success
* independence of joint time interval (=> same as saying in discrete time that different time are indep r.v.)
* during very small delta time (infinitely small). 
    - proba of seeing 1 client is lambda * delta
    - not seeing is 1-lambda*delta
    - ore than 1 client is 0 
    - Lambda is "arrival rate" and is (expected # arrival)/time unit
* number of arrival in interval of lentgh tau (was binomial in discrete case). can approx as having n bernoulli where n = tau/delta and proba of each is p=lambda * delta => take limit of bernoulli when delta -> 0 => poisson
* interval time distribution is limit of geometric as delta-> 0 => exponential


info : 
* https://www.youtube.com/watch?v=jsqSScywvMc&index=14&list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8
* https://www.youtube.com/watch?v=XsYXACeIklU&list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8&index=15

### Markov Chain

* generalization of previous processes but now not indep
* new state = f(old state, noise)
* Pretty much any process can be described as markov process if defined with the correct notion of state

info:
* https://www.youtube.com/watch?v=IkbkEtOOC1Y&list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8&index=16

### Simple symmetric random walk

Dsicrete stochastic process where the location can only jump to neighbouring sites of a lattice (Simple) with equal probability (symmetric). *e.g.* often talk about random walk where can only take a step +1 or -1 whith Bernoulli probability.  

Notes:
- maximum height achieved is on the order of sqrt(n)


### Brownian Motion 
Continuous time generalization of a random walk. 

definition:
- B(0) = 0
- B(t) indep of B(t')
- B(t) - B(s) ~ gaussian(0, t-s)




Note:
- it is the integral of a white noise gaussian process (i.e goes up or down with mean 0), but brownian adds this noise to last value => integral
- crosses the time axis infinitely often
- most often it's contained in [-sqrt(t), sqrt(t)]
- nowhere differentiable => cannot use calculus (have to use Ito's calculus that extends calculus to this)
- one genralization of Brownian motion is an **ito process**, which basically is a brownian motion with possibly non zero mean (=> introduces correlation)
- an other generalization is the **marrtingale**Stochastic process where $$\mathbb{E}[X_{n+1}\vert X_{1:n-1}]=X_n$$. It is a generalization of a random walk and brownian motion.
- use langevin equation (f=ma generalized when random forces), to study systems of brownian motion
- also a gaussian process with mean 0 and variance t and cov(s,t) = s (becaus indep => only variance) 

### Gaussian Process
Collection of r.v. s.t. every finite subset if distribution accoring to multivariate gaussian


## Statistics
### Definitions
**Nuisance Parameters**
Any parameter which is not of immediate interest but which must be accounted for in the analysis of those parameters which are of interest. The classic example of a nuisance parameter is the variance, σ2, of a normal distribution, when the mean, μ, is of primary interest. Usually you have to marginalize over the nuisance parameters.
**Statistic**
More generally, we call a function of the data, say, T = t(Xn
) a statistic. A statistic is referred to as sufficient for the parameter θ, if the expression P(Xn ∣T = t, θ) does not depend on θ itself. T
**Moments**
n-th moment of a function: 
$$\mu_n = \int_{-\infty}^\infty (x - c)^n\,f(x)\,\mathrm{d}x$$

=> for probability E[(X-c)^n]. c=0 => raw moment moment. c = mean => central moemnt, they describe the sahpe of the function independently of translation.

FOr PDF:
- Normalized nth central moments:  E[(X-mu)^n]/sigma^n
- mean : first moment 
- variance : second central moment
- skewness : third normlized moment
- kurtosis _ fourth normalized moment

## Linear Algebra 
## Math 
### General
#### Norm

##### Finite Dimensions

Property:
- $\vert \mathbf{x} \vert \geq 0$ and equality only if $\mathbf{x} = 0$
- (positive homogoneity) $\vert \alpha x \vert = \alpha \vert  x \vert, \ \forall \alpha \in \mathbb{R}$
- triangle inequality 

These are true for all $L_p$ norms with $p \geq 1$ :

$$||\mathbf{x}||_p = (sum_{-\infty}^{\infty} |\mathbf{x}|^p )^{1/p}$$

Important norms:
- l1
- l2
- l inf

Note:
- $\left\| x \right\| _p \leq \left\| x \right\| _r \leq n^{ (1/r - 1/p) } \left\| x \right\| _p, \ forall 0 < r < p$
- can be extended to infintie number of components (sequence space). This yields the space $ℓ<sup>&thinsp;p</sup>$
- can be extended to function for which  the $p^{th}$ power of the absolute value is integrable : $L<sup>p</sup>$ space (normally lebesgue integrable but four our use cases we can use riemanian integrals):

$$||f||_p (∫_{-\infty}^{\infty} |f(x)|^p dx)^{1/p} < \infty$$


#### Hilbert Space
- geenralization of euclidean space
-  complex or real inner product space (i.e.  vector space with an inner product) which is complete (i.e. any sequence of points that get closer and closer to each other converges to some point) with respect to the distance (norm between points that satisfies triangular ineq.) defined by the inner prod

Notes : 
- generalization of R^n

Ex:
- L2 space: space of square integrable function $||f||_2 < \infty$ 

Info: 
-https://www.quora.com/What-are-Hilbert-Spaces-in-laymens-terms
-https://sadeepj.blogspot.com/2014/05/a-tutorial-introduction-to-reproducing.html



### Measure
$\mu$ is intuitively the size of a set. Example measurre of a box is area. 
mu(empty) = 0, sum of measure union is measure of sum of union in disjoint set, measure of subset is smaller than measure of set, 

ex:
- lebesgue measure smallest possible sum of open intervals such that the union of the intervals cover the set we are measuring E. Which is what you would think of, for example lebesgue of segment $\[a,b\]$, you take all sets open intervals that cover the segment, and take the minimum of the sums => b-a




*Nota Bene: these terms are not always the most important ones but important ones I have encountered since my "migration" to machine learning / computer science in September 2016.*

Thanks to [Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/), my Machine Learning professor, who introduced me to this amazing field.



-----
Additional ----
#### Discriminative Classifiers

I devoted a [section](#generative-vs-discriminative) to discriminative classifiers but in summary these are the algorithms that directly learn a (decision) boundary between classes.

As a reminder these can be either:
* **Probabilistic**: the algorithm has a probabilistic interpretation: it tries to model $p(y\|x)$.
* **Non-Probabilistic**: the model cannot me interpreted with probabilities: it simply "draws" a boundary which you will simply use to predict. 


-----

Other concepts ??
- Maximum Mean Discrepancy 
