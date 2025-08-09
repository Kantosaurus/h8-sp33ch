# TASK 3: Hate Speech Classification Project

## Overview
**Hate Speech Classification – 50-007 Machine Learning (Summer 2025)**

This team-based project challenges you to develop machine learning models that detect hate speech in social media posts. You'll approach the problem as a binary text classification task, applying a mix of foundational techniques and custom models.

Your work will be assessed through multiple tasks, including algorithm implementation, evaluation of model performance, and submission to the class leaderboard. A strong emphasis is placed on reproducibility, creativity, and clarity in your documentation.

Check the Tasks section below for detailed instructions and deliverables. Your submissions will contribute both to your individual grade and to your team's position on the leaderboard.

**Good luck, and build responsibly.**

## Goal
The goal of your project is to develop and evaluate machine learning models that can detect hate speech in social media posts. It's framed as a binary classification problem—predicting whether a given post is hateful or non-hateful.

## Description
This is a team project. You are encouraged to form teams in any way you like, but each team must consist of either 4 or 5 people.

### Project Summary
Online hate speech is an important issue that breaks the cohesiveness of online social communities and even raises public safety concerns in our societies. Motivated by this rising issue, researchers have developed many traditional machine learning and deep learning methods to detect hate speech on online social platforms automatically.

Essentially, the detection of online hate speech can be formulated as a text classification task: "Given a social media post, classify if the post is hateful or non-hateful". In this project, you are to apply machine learning approaches to perform hate speech classification. Specifically, you will need to perform the following tasks.

## Task 3: Try other machine learning models and race to the top! (25 marks)

In this course, you are exposed to many other machine learning models. For this task, you can apply any other machine learning models (taught in the course or not) to improve the hate speech classification performance! Nevertheless, you are **NOT TO use any deep learning approach** (if you are keen on deep learning, please sign up for the Deep Learning course! - highly encourage!).

To make this task fun, we will have a race to the top! Bonus marks will be awarded as follows:

- **1 mark**: For the third-highest score on the private leaderboard.
- **2 marks**: For the second-highest score on the private leaderboard.
- **3 marks**: For the top-highest score on the private leaderboard.

Note that the private leaderboard will only be released after the project submission. The top 3 teams will present their solution on week 13 to get the bonus marks!

### Key Task Deliverables

**3a.** Code implementation of all the models that you have tried. Please include comments on your implementation (i.e., tell us the models you have used and list the key hyperparameter settings).

**3b.** Submit your predicted labels for the test set to Kaggle. You will be able to see your model performance on the public leaderboard. Please make your submission under your registered team name! We will award the points according to the ranking of the registered team name.

## Task 4: Documenting your journey and thoughts (5 marks)

All good projects must come to an end. You will need to document your machine learning journey in your final report. Specifically, please include the following in your report:

- An introduction of your best performing model (how it works)
- How did you "tune" the model. Discuss the parameters that you have used and the different parameters that you have tried before arriving at the best results.
- Did you self-learned anything that is beyond the course? If yes, what are they, and do you think if it should be taught in future Machine Learning courses.

### Key Task Deliverables
**4a.** A final report (in PDF) answering the above questions.

## Submission

All outputs, including the final report, must be zipped and uploaded to e-Dimension. Only submissions uploaded to e-Dimension will be marked. One member from each group is required to submit the zipped file under the assigned group number. E-mail submissions are not acceptable.

## How Your Kaggle Leaderboard Score Is Calculated

### Task Overview
Your task is to:
- Train a model using `train.csv`
- Predict labels for data in `test.csv`
- Submit your predictions via `submission.csv`

### What Does the Leaderboard Score Mean?
Your Kaggle score is based on **Classification Accuracy**, computed as:

```
Accuracy = (Number of Correct Predictions) / (Total Predictions)
```

**Example:**
If you correctly predict 3,800 out of 4,296 rows, your leaderboard score will be:
`3800 / 4296 ≈ 0.8848`

### File Structure Expectations
Your `submission.csv` must:
- Have two columns: `row ID` and `label`
- Match the structure of the sample:

```csv
row ID,label
17185,0
17186,1
...
```

A validated example file is: `submission1.csv`

### What Kaggle Actually Does Internally
1. Instructor uploads a hidden `solution.csv` (ground truth for `test.csv`)
2. You upload your `submission.csv`
3. Kaggle runs the evaluation script using `sklearn.metrics.accuracy_score`
4. Your score reflects how many predictions match the true labels

### Score Is NOT Based On:
- Training accuracy
- Loss or precision/recall
- Your model's complexity
- Your `train.csv` performance

### Submission Tips
- Use the same structure as `sample_submission.csv`
- Submit predictions for all test IDs
- Do not include extra columns or leave rows missing
- Save as UTF-8 CSV without BOM (e.g., use Notepad or Excel → Save As → CSV UTF-8)

## Acknowledgements

This dataset is from an awesome group of researchers from UC San Diego and Georgia Institute of Technology.

ElSherief, M., Ziems, C., Muchlinski, D., Anupindi, V., Seybolt, J., De Choudhury, M., & Yang, D. (2021, November). Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 345-363)
