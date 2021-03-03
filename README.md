# Automatic Scoring for Picture Description
Automatic grading system for language learners.

## About
In language learning, training output skill such as speaking and writing is vital in order to retain the learned knowledge. However, scoring descriptive questions by humans would be costly, and this is why automatic scoring systems attract attention. In this research, we try to realize an automatic scoring system for picture description. Concretely, (i) we first analyze the trends of errors that English learners would make, (ii) then create a pseudo dataset by artificially mimicking the errors, and (iii) finally consider a model that judges whether a given pair of a picture and a sentence is valid or not. In experiments, we trained the model with the created pseudo data and evaluate it with the answers provided by actual learners. From experimental results, we found that our model outperforms a random agent.

## Picture Description 
In this task, language learners are asked to describe a picture.
- Write 1 sentence that is based on a picture.
- With each picture, you will be given 2 words or phrases that you must use in your sentence.
- You can change the forms of the words and you can use the words in any order.


## Model
Judge the correctness of the answer based on the semantic features of the picture and answer.


