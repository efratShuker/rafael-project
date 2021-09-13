# rafael-project

The subject of this project is classification rockets

* First I known the data like:
Display amount of routes for each type
Display histograms of length of routes rockets for each type:

![image](https://user-images.githubusercontent.com/86189441/133099751-a970043b-4f39-4128-8723-4c01e981eb95.png)

* Drawing  

I want to see different between the type 1 and 16, so I draw gragh:
- draw the first route:

![image](https://user-images.githubusercontent.com/86189441/133099847-1e008661-c65c-459b-a9de-037cc5a07cdb.png)

- draw 50 first routes with type 1, 2, 3, 4, 5, 6:

![image](https://user-images.githubusercontent.com/86189441/133099886-956da13f-03ef-41d2-ba2e-5a5ea6cd02c5.png)

 
- draw 50 first routes with type 1 and 6 with length of routes 15 second. In order to know length how many second have to routes I add to data new column with name "Length" this column with type int that say the length of route in seconds. Now to draw 50 first routes with type 1 and 6 with length of routes 15 second I check for every routes if "Length" is 15.
See in drawing type 1 with color blue and type 16 with color yellow

![image](https://user-images.githubusercontent.com/86189441/133099921-2d70f9bd-8acf-47f3-8bd3-8444d695d265.png)
 
in this drawing you can see the different between type 1 and type 16, the length is different and type 16 is higher. I built the intuitive rules according this.

* Making data
I create two new tables: split the data to two parts. One is 20% from data and second is 80% from data and create from it new tables, the tables only types 1 and 16
The table with 80% is train table in order to train the model with data for recognize if type is 1 or 16. And table with 20% is for checking the model, so I delete the column of types from check set.
*classification with intuitive rules
First I draw all rockets with type 1 and 16 to see the different:

![image](https://user-images.githubusercontent.com/86189441/133099958-e4c9b963-b04b-4a27-877a-de4cb4a2f439.png)

Now after I have draws and two table one train set and second check set I built intuitive rules to recognize if type is 1 or 16
The intuitive rules :
- like you can see in last drawing, length of routes of rocket with type small than 5000, and high of rocket small than 7500, else type is 16.
I calculate confusion matrix in order to see percents of success:
[[331   3], 
 [55   79]]
 And calculate f1 score: 0.9195
The result is not very good Its only 91%

Now in order to improve the result I calculate energy for all routes with type 1 and 16:

![image](https://user-images.githubusercontent.com/86189441/133099984-311fb1c5-d6c1-42bf-82eb-4e96112c896c.png)
 
I add to intuitive rules validation of energy, I calculate energy for end of routes and check if it small than 155000 then type is 1 else type is 16.
I calculate the confusion matrix:
[[317     0]
 [0     151]]  
And calculate f1 score: 1.0
Now you can see that result is 100% !!!!

*machine learning
I try RandomForestClassifier:
The confusion matrix:


[[ 311  3   2   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   1   4   1   4 143]]
Average Error: 0.0700 degrees
And accuracy: 0.968017
Result is 96%

I try LogisticRegression
The confusion matrix:
[[316     0]
 [0     153]]  
Average Error: 0.0000 degrees
And accuracy: 1.0
Result is 100%
This is very good to classification rockets with types 1 and 16!!
* classification rockets with type 1, 4, 7, 10
I create two new tables: split the data to two parts. One is 20% from data and second is 80% from data and create from it new tables, the tables only with types 1, 4, 7, 10
The table with 80% is train table in order to train the model with data for recognize if type is 1, 4, 7, 10. And table with 20% is for checking the model, so I delete the column of types from check set.
*classification with intuitive rules
I draw the routes of rockets with types 1, 4, 7, 10:

![image](https://user-images.githubusercontent.com/86189441/133100043-4018c9d3-1839-43eb-85da-b3b44ad9365b.png)
 
I draw the calculate energy for each type:

![image](https://user-images.githubusercontent.com/86189441/133100059-5ba5c5df-6da8-4bd6-b0b2-d4f570f0839b.png)

And according the drawings I decide which type it is:
intuitive rules is validation of energy, I calculate energy for end of routes and check if it bigger than 260000 or distance between first second for last second of routes bigger than 18160 then type is 10.
Else if energy of end second bigger than 162500 or distance between first second for last second of routes bigger than 12176 then type is 7.
Else if energy of end second bigger than 106000 or distance between first second for last second of routes bigger than 7365 then type is 4.
Else type is 1.
I calculate confusion matrix:

[[334   0      0     0]
 [128   128    3     0]
 [15    143   161    0]
 [23    49    119   56]]

And f1 score: 0.5858498705780846
Result is 58.6% is not good

*machine learning
I try LogisticRegression
The confusion matrix:
[[317  0     0    0]
 [1  179    71    8]
 [1   25   274   48]
 [0   15    19  202]] 

Average Error: 0.5483 degrees
And accuracy: 0.8379310344827586
Result is 83.8%
It is good but not excellent

I try RandomForestClassifier
The confusion matrix:
[[317   0    0    0]
 [0   257    2    0]
 [0     3   329  16]
 [0     3    29  204]] 
Average Error: 0.1448 degrees
Accuracy: 0.9543103448275863
Result is 95.4% it is very good

I try RandomForestClassifier with default values:
Average Error: 0.1526 degrees
Accuracy: 0.9517241379310345
Result is 95.2% it is not better than last

I try RandomForestClassifier with default values but I check on train set instead test set:
Average Error: 0.0000 degrees
Accuracy: 1.0
And result is 100%

I try to change values of number of tree and depth of tree in RandomForestClassifier

![image](https://user-images.githubusercontent.com/86189441/133100403-b21e6e8f-5982-4f83-9ef8-714826579835.png)


You can see that 1000 tree and 50 depth is best: 95.87%
I see that need number of tree bigger than depth of tree.
