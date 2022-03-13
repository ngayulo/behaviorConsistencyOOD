# Evaluating Object Recognition Behavior Consistency given Out-of-Distribution Stimuli

Based on experiments done by [Geirhos et al 2020][1], this project set out to evaluate the ability for CNN to predict human behaviors given out-of-distribution stimuli.  
We show that using a trained decoder and the Image1 metric to measure consistency, consistency between humans and CNN's is not at chance as suggested in Geirhos et al. 
# File outline
- **codes** contains most python codes used in experiments
- **model-results** contains all files relating to their activations (contained in **same domain**), and the predicted responses of the ANN models.
- **subject-results** contains all files relating to human predictions on the stimuli presented.

# Data
All human data and images were taken from Geirhos et al. We used four types of images, the original, silhouette, edges, and cue-conflict. 

# Files and Experiments
General outline of the experiments go as follow:
**features.py** and **feature_extraction.py** extract the layer activations from the penultimate layer of an ANN model given the original styled images. The activations are used to train a logistic regression decoder written in **regression.py**. We train ten decoders in total for each CNN model. The out-of-distribution (o.o.d.) images (silhouette, edges, cue-conflict)  are given to the trained decoder. At this point, we consider two methods of measuring human consistency. We use the kappa statistics from Geirhos et al and the Image1 metric from [Rajalingham and Issa et al][2]. Functions used for these metrics is found in **behavior_metric.py**. 

- **train_model.py** is used to get the trained decoders
- **train_model_cross.py** takes the trained decoders and input o.o.d. images to get responses
- **train_model_wog.py** trains decoders with some percentage of the o.o.d. images.
- **geirhos.py** replicates the experiment in Geirhos et al for the model's raw responses.

# Acknowledgement
This project was done under the mentorship of Tiago Marques and the DiCarlo Lab at MIT CBMM. It was funded by MIT MSRP-Bio. 

[1]: https://papers.neurips.cc/paper/2020/file/9f6992966d4c363ea0162a056cb45fe5-Paper.pdf
[2]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6096043/





