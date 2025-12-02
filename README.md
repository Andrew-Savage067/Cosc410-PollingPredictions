# Cosc410-PollingPredictions
Final Project for Cosc410 - Using Machine learning to predict election results based on polling data

Got data from:
Kotak, A., & Moore, D. A. (2022). Election polls are 95% confident but only 60% accurate. Behavioral Science & Policy, 8(2), 1-12. https://doi.org/10.1177/237946152200800202 (Original work published 2022)

Used  decision tree classifier fron SKlearn to predict polling results based on days before the election, and the polling numbers of each candidate.
In this simplifie model ive removed data about polling agency which will likely be used in the final project.

Another change that will need to be made is that the time component is not really feasible in the decision tree so ive just taken my test data to be random polls
in the data set with the correct answer alwaysa being the final reslts of the poll on election day.

The results are about what id expect, it guesses the correct winner some of the time with its mean accuracy being in the range of 50% - 80% depending on the random value used which is a good sign as it offers much room for improvement.
