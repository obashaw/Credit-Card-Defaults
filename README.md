This was an independent project that I worked on during the Summer of 2019.  I am
self taught in Python, with no formal instruction.  I used Scikit Learn's reference
page to help with the implementation and also read a couple of blog posts & articles.

I started with the SVC model and read up on parameter tuning until I found out about
GridSearchCV.  Understanding the C and gamma parameters and different kernel types
was helpful, and I eventually got my raw accuracy to the low 80's.  I did notice a 
substantial number of Type II (false negative) errors, which is problematic in this
case because it deals with money.

Then I moved to the Decision Tree Classifier, which was an easier implementation.  
It seemed too simple at first, but when I grew the number of leafs and branches on the 
tree, I realized that I was severely overfitting.  So, I limited it to two levels (three 
nodes total) and was again able to achieve low 80's accuracy with a simple model.

I wanted to find a way to get the probability of a client defaulting vs paying their 
debt, so SVR was the third and final model. I first had some trouble with a lack of 
differentiation in the answers (a ton of test predictions with the same probability 
for each class), and the realized that my gamma value was way too high.  I decreased 
it, and acheived a .score() of 86-87%.
