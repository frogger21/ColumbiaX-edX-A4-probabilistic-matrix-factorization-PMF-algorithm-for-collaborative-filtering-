Probabilistic Matrix Factorization PMF algo for collaborative filtering.

![alt text](https://github.com/frogger21/ColumbiaX-edX-A4-probabilistic-matrix-factorization-PMF-algorithm-for-collaborative-filtering-/blob/master/edx5.JPG)

We have a matrix of movie reviews with N1 users (on the rows) and N2 movies (on the columns). As it is unlikely that every user saw every movie, we will have many missing data. Via matrix factorization we hope to estimate the missing movie reviews with the matrices: U and V. Similar movies should have similar rating patterns and their v's should be close to each other in the R^D dimension.
Users with similar preferences in movies should also have their u's close to each in other in R^D.

The PMF algorithm done in python uses coordinate ascent to estimate the matrices U and V so that we can estimate the missing movie reviews in the data with the dot product of row i of U and row j of V, i.e. the user i rates movie j with u(i)'v(j). 
