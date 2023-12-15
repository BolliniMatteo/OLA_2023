# OLA Project (Pricing-Advertising)

Python codebase of the final project for the Online Learning Applications course (A.Y. 2022/2023) @ Politecnico di Milano.

## The team (#14)

| Name and surname                | Email                                  |
| :------------------------------ | :------------------------------------- |
| Matteo Bollini                  | matteo3.bollini@mail.polimi.it         |
| Michele Guerrini                | michele.guerrini@mail.polimi.it        |
| Davide Mozzi                    | davide.mozzi@mail.polimi.it            |
| Carlos Alberto Santillán Moreno | carlosalberto.santillan@mail.polimi.it |
| Davide Tonsi                    | davide.tonsi@mail.polimi.it            |

## Problem description

The task is to optimize both the pricing and advertising strategies related to the sale of a specific product by an e-commerce to maximize profit. In particular the focus is on finding the optimal price and advertising bid between two sets of candidates, while minimizing the cumulative regret.

## Run the experiments

Here are all the steps required by the project proposal with thei respective scripts to run to get the plots of the results:

| Step | Description                                                       | Python script                                       |
| :--- | :---------------------------------------------------------------- | :-------------------------------------------------- |
| 1    | Learning for pricing                                              | `Step1Experiment.py`                                |
| 2    | Learning for advertising                                          | `Step2Experiment.py`                                |
| 3    | Learning for joint pricing and advertising                        | `Step3Experiment.py`                                |
| 4    | Contexts and their generation                                     | `Step4TSExperiment.py`, `Step4UCBExperiment.py`     |
| 5    | Dealing with non-stationary environments with two abrupt changes  | `Step5Experiment.py`                                |
| 6    | Dealing with non-stationary environments with many abrupt changes | `Step6Experiment.py`, `Step6Experiment_3_phases.py` |

Some other scripts that are useful to create plots that are in the report:

| Reference | Description           | Python script         |
| :-------- | :-------------------- | :-------------------- |
| Slide 6   | Class curves          | `class_curves.py`     |
| Slide 15  | Step 1 value function | `step1_analysis.py`   |
| Slide 46  | Curves to optimize    | `analyze_classes.py`  |
| Slide 53  | Step 5 phases         | `analyze_3_phases.py` |
| Slide 65  | Step 6 phases         | `analyze_phases.py`   |

## Requirements

Python version: `3.11.1`

Essential Python Packages:
```
matplotlib==3.8.0rc1
numpy==1.25.2
scikit-learn==1.3.0
scipy==1.11.2
tqdm==4.66.1
```
