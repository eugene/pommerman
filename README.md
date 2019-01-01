# Pommerman
PyTorch based solution for the Pommerman competitions (https://www.pommerman.com/) done as an exam project in course 02456 - Deep learning (http://kurser.dtu.dk/course/02456) at DTU - Technical University of Denmark.

## Requirements 
In addition to `pytorch` (https://pytorch.org) and the usual data science packages (`numpy`, `matplotlib`) this project depends on the Pommerman `playground` (https://github.com/MultiAgentLearning/playground) to be installed in your Python environment.

## Usage
`python main.py zzzzzzz`

### A3C Model

#### Immitation learning
#### Training
#### Evaluating

### A2C Model
Run the following:

To generate the weights file (refreshed every 300 episodes):
```python A2C/main.py train``` 

To see how your agent plays (loads the weights and can be used while the training is running)
```python A2C/main.py eval``` 

During the training current `gamma`, `running reward`, `action statistics` and `loss` are printed after each episode. It takes around 48 hours to fully train this model (40000 episodes) on a modern 10 core CPU with a single 1080TI GPU. Additionally, a `training.txt` file is generated with the main statistics for each trained episode.

## Main results
Business model canvas interaction design partner network pitch business plan. Business-to-consumer accelerator startup partnership validation ramen network effects branding metrics monetization mass market. Beta bandwidth twitter facebook seed money innovator vesting period social proof pivot.

## Paper
Business model canvas interaction design partner network pitch business plan. Business-to-consumer accelerator startup partnership validation ramen network effects branding metrics monetization mass market. Beta bandwidth twitter facebook seed money innovator vesting period social proof pivot.

## Credits
* Kris Walther (s172990, https://github.com/KrisWalther)
* Mirza Hasanbasic (s172987, https://github.com/kazyka)
* Søren Hjøllund Jensen (s123669, https://github.com/SorenJ89)
* Yevgen Zainchkovskyy (s062870, https://github.com/eugene)