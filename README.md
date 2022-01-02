# Convex Clustering
* Objective
<img src="https://latex.codecogs.com/svg.image?\min_{\mathbf{X}&space;\in&space;\mathbb{R}^{d\times&space;n}}&space;f_{clust}(\mathbf{X})&space;=&space;\frac{1}{2}\left\|&space;\mathbf{X_i}-\mathbf{A_i}\right\|^2&space;&plus;&space;\lambda&space;\sum_{i=1}^{n}\sum_{j=i&plus;1}^{n}\varphi_{hub}(\mathbf{X_i}-\mathbf{X_j})" title="\min_{\mathbf{X} \in \mathbb{R}^{d\times n}} f_{clust}(\mathbf{X}) = \frac{1}{2}\left\| \mathbf{X_i}-\mathbf{A_i}\right\|^2 + \lambda \sum_{i=1}^{n}\sum_{j=i+1}^{n}\varphi_{hub}(\mathbf{X_i}-\mathbf{X_j})" />

where <img src="https://latex.codecogs.com/svg.image?\varphi_{hub}(\mathbf{y})=\begin{cases}\frac{1}{2\delta}\left\|&space;\mathbf{y}\right\|^2&space;&&space;\text{&space;if&space;}&space;\left\|&space;\mathbf{y}&space;\right\|\leq&space;\delta&space;\\\left\|&space;\mathbf{y}&space;\right\|&space;-&space;\frac{\delta}{2}&&space;\text{&space;if&space;}&space;\left\|&space;\mathbf{y}&space;\right\|>&space;\delta&space;\end{cases}&space;" title="\varphi_{hub}(\mathbf{y})=\begin{cases}\frac{1}{2\delta}\left\| \mathbf{y}\right\|^2 & \text{ if } \left\| \mathbf{y} \right\|\leq \delta \\\left\| \mathbf{y} \right\| - \frac{\delta}{2}& \text{ if } \left\| \mathbf{y} \right\|> \delta \end{cases} " />

**X** and **A** are the parameter and data matrix, respectively. d is the number of features and n is the number of points we want to cluster. Each column of **A** is a data point and each column of **X** is the "centroid" of each data point. After solving this equation and get the optimal solution for **X**, different data points can be clustered in the same group if 

<img src="https://latex.codecogs.com/svg.image?\left\|&space;\mathbf{x_{i}^{*}}&space;-&space;\mathbf{x_{i}^{*}}\right\|&space;<&space;\varepsilon&space;" title="\left\| \mathbf{x_{i}^{*}} - \mathbf{x_{i}^{*}}\right\| < \varepsilon " />

for some predefined hyperparameter <img src="https://latex.codecogs.com/svg.image?\varepsilon&space;" title="\varepsilon " />.
