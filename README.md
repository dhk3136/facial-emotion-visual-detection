# Human facial emotion detection - a seven-category CV classifier

### Overview
Human emotion detection is a downstream task of Computer Vision. Classification of human emotions remains a difficult endeavor for ML and CV practitioners despite major advances in both network architectures and algorithms. Just a couple of years ago, SOTA scores hovered around the 0.37 accuracy mark.

### Purpose
This project attempts to build upon past work to achieve higher marks in accuracy using a keras CNN implementation. The task is to accurately classify seven different emotions from the given images: Angry, Sad, Happy, Fear, Disgust, Surprise, and

### Data
I use the FER dataset featured in "Challenges in Representation Learning: A report on three machine learning contests," authored by well-known GAN innovator, Ian Goodfellow. The dataset features actors in various states of facial expression which are to be classified into the categories denoted above.  

### Analysis and Results
My best results achieved an accuracy of 0.58 which is quite an improvement over scores from when the dataset was initially released. Training was conducted between a range of 5 epochs to 100 epochs. Despite the vast difference in the length of training runs, results did not vary much: at most, a 0.07 difference was observed based on the number of epochs.

Overfitting was a major problem. Typically, in multi-class categorizations, annotators are given permission to tag more than one category which in turn creates multicollinearity in the data.

After fine-tuning hyperparameters, I was able to fit training data much more closely to validation sets and reduce overfitting. 15 epochs seemed to be the sweet spot for running inference. In addition I introduced another round of layers (dropout, dense) and changed batch sizes in a range between 128-1024. This tuning greatly improved performance.



