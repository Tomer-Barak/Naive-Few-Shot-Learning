# Naive Few-Shot Learning

Replicate the main results from the paper Naive Few-Shot Learning: Sequence Consistency Evaluation. The main.py code creates an SCE test and solves it using the Markov-CPC model. This can be iterated for as many SCE tests as necessary by changing "N_tests".

A test is generated according to a dictionary of the form {"color": 1, "position": 0, "size": 2, "shape": 0, "number": 0}. The features with integer 0 remain constant throughout the test, those with 1 are the distracting features which change randomly, and the feature with 2 is the predictive feature that changes according to a simple deterministic rule.


Instructions:
1. Open main.py
2. Change the hyperparameters dictionary (HP) according to your interests.
3. Run.
