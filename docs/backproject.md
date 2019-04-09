---
title: 3D features maps
header-includes: |
numbersections: true
---

# The model

Let $K$ denote the intrinsic matrix of the camera. Let $N$ denote the number of pictures taken by the scanner, and
$R_i, T_i$ denote the extrinsics parameters of the camera for the $i$th shot,
$i$ ranging from $1$ to $N$. For any point $x\in\mathbb{R}^3$ in the world coordinates, the projection of $x$ onto the
$i$th shot is given, in homogeneous coordinates, by $Y = K(R_i x + T_i)$. If 

$$
Y = \left( \begin{array}{ccc}
    x \\
    y \\
    z
    \end{array} \right),
$$

then $P_i(X) = (x/z, y/z)$ is the projection in pixel coordinates of point $X$. 

Let $\Omega = [0, W] \times [0, H]$ be the domain of each image, where
$W$ and $H$ are the width and height of the image in pixel respectively.

Then, an image is viewed as a function

$$
I_i : \Omega  \rightarrow \mathbb{R}^3
$$

# 2D features

just to make things a bit more concrete, I will list a list of features that we can use on 3D pictures

## Color or geometry based features

## Learned semantic segmentation

The thi

# Backprojections

From the previous section, we obtain, for each 


$$M_i : \mathbb{R}^2 \rightarrow \mathbb{R}^d$$

is the feature map corresponding to picture $i$. $d$ is the number of features. For example,
in the case of Vesselness features, $d=1$. If 2D features correspond to multi-class semantic
segmentation, $d$ is the number of classes. We assume that each feature map is normalized in the
$[0,1]$:

$$M_i(x, y)_k \in [0,1], \forall (x,y)\in\mathbb{R}^2, \forall i\in\{1,\cdots, N\}, \forall k \in \{ 1,\cdots, d \};$$
$$\sum_{k=1}^d M_i(x, y)_k = 1, \forall (x,y) \in \mathbb{R}^2. $$

Therefore, we can see $M_i(x, y)_k$ as the probability that, given picture $I_i$, pixel $(x,y)$ is the projection of a point
belonging to class $k$.

## Visual hull

We present the visual hull, also called the \emph{space carving} method to obtain a 3D volume from the 2D feature maps. To
compute the 3D volume corresponding to class $k \in \{1,\cdots,d\}$, we estimate the label of a pixel $(x,y)$:

$$L_i(x,y) = \textrm{argmax}_k M_i(x, y)_k.$$

Then, for each class $k \in \{1,\cdots,d\}$, the visual hull of class $k$ is the set of all points which project only on pixels
labels as class $k$:

$$V_k = \{ X \in \mathbb{R}^3: L_i(P_i(X)) = k, \forall i \in \{ 1,\cdots, N\} \}.$$

Since each pixel can only be in one class, the major drawback is that any occlusion will create a smaller visual hull. It is 
therefore only realistic to use this with 2 classes, like in the case of an object that we segment from the background. The method
is also very senstivie to false negative and wrong camera poses.

## Independent backprojections

If we make the simplifying assumption, that projections are independent - they are not - and that 
the feature $M_i(x,y)_k$ represent the probability that the ray issued fromp pixel $x,y$ in camera $i$
a point of class $k$, then 

$$ \log(P(L(X) = k) = \sum \log P(k \in L_i(P_i(X))), $$

where $L$ is the mapping from $\mathbb{R}^3$ to the label of the corresponding point.

