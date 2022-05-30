# MT_Russian-to-RSL
MA Thesis by Anna Klezovich

This project is devided in three parts:

- In the first part I was building text-to-gloss machine translation baseline. It was a sequence-to-sequence textbook model with attention and "learning to sample" mechanism trained on the original data from scratch. Then there was an experiment with finetuning russian T5 model on the original, which showed better score and became a baseline.
- In the second part of the experiment I conducted experiments with three data augmentation techniques and their mixtures in different proportions. Then I compared which augmentation technique was better.
- Last but not the least I explored the figure\ground relationship in the classifiers on the test dataset. However, the dataset was too small to draw concrete conclusions in this part, except for the fact that all spatial constructions in sign languages should be trnaslated with the help of computer vision, because they are annotated inconsistently and show a lot of unique to the situation patterns (aka. they cannot be lexicalized).
