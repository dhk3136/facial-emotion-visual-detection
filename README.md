# Human facial emotion detection - a seven-category CV classifier

### Overview
Human emotion detection is a downstream task of Computer Vision. Classification of human emotions remains a difficult endeavor for ML and CV practitioners despite major advances in both network architectures and algorithms. Just a few years ago, SOTA scores hovered around the 0.37 accuracy mark.

### Purpose
This project attempts to build upon past work to achieve higher marks in accuracy using a keras CNN implementation. The task is to accurately classify seven different emotions from the given images: Angry, Sad, Happy, Fear, Disgust, Surprise, and Neutral.

### Data
I use the [FER dataset](https://arxiv.org/abs/1307.0414) featured in "Challenges in Representation Learning: A report on three machine learning contests," authored by well-known GAN innovator, Ian Goodfellow. The dataset features actors in various states of facial expression which are to be classified into the categories denoted above.

The data was divided into train/test sets of 28709 and 3589 expressions, respectively.  

### Analysis and Results

![dee](img/dee_256.png)
![plot_dee](img/plot_dee.png)
> Kandyse McClure, better known as "Dee" in the remake of Battlestar Galactica  

My best results achieved an accuracy of 0.58 which is quite an improvement over scores from when the dataset was initially released--although I concede I was hoping for a better improvement, overall. Training was conducted between a range of 5 epochs to 100 epochs. Despite the vast difference in the length of training runs, results did not vary much: at most, a 0.07 difference was observed based on the number of epochs.

![kate_winslet](img/kate_winslet_256.png)
![plot_kate_winslet](img/plot_kate_winslet.png) 
> Kate Winslet shoots an ambiguous expression across the car  

Overfitting was a major problem. Typically, in multi-class categorizations, annotators are given permission to tag more than one category which in turn creates multicollinearity in the data. For example, it's possible for an expression to be considered as Neutral, Angry, and Sad (to be fair, at differing probabilities).  

After fine-tuning hyperparameters, I was able to fit training data much more closely to validation sets and reduce overfitting. 15 epochs seemed to be the sweet spot for running inference. In addition, I introduced another round of layers (dropout, dense) and changed batch sizes in a range between 128-1024. This tuning greatly improved performance.

### Conclusion
Given the real-life ambiguities of human emotions, it's more than understandable that classes would not have sharp definition into one category but rather bleed into one another. This problem is the single biggest hurdle of emotion classification: it depends upon annotators' judgements related to distinctions of emotions that perhaps are not always meant to be distinguished.  

Yet more work can be done to make both annotation and practitioner analysis more sophisticated. For example, a time-series analysis could turn the classification categories into a distribution of semantic similarities based in time-space rather than more stagnant, 'snapshot'-based classification. In fact, attempts at that work is already beginning with real-time, video and web-cam based classifications, although those analyses also return to the single-image classification metric with the primary difference being one is a moving set of images while the other is not. Perhaps, when time is considered the crucial distribution first, better signal can be captured as emotion classification begins to articulate with time-series distributions.  
