# LT2212 V20 Assignment 3

Part1

Run a3_features.py in the terminal as follows:

python3 a3_features.py enron_sample outputfile.csv 1914

where: 
enron_sample(inputdir) = The input directory containing the data
outputfile.csv(outputfile) = The csv output file with the feature table

1914 (dims) -The output feature dimensions.

there is a 5th argument (--test) = The percentage (integer) of instances to label as test.
set as default to 20%


Part2

To be ran in the terminallike this:
a3_model.py outputfile.csv

I have completed the assignment by following different examples, including that of trimodule.py. It took me a while to figure out the sampling process, and it was a challenge to design the code in such a way that different functions could be used in a relatively efficient manner.But overall, I have learnt many new things while working on it.

The results are not that good, just as expected, most probably due to class imbalance.
