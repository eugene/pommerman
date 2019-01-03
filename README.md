# Pommerman :bomb:
PyTorch based, reinforcement learning solution for the [Pommerman competitions](https://www.pommerman.com/) done as an exam project in course [02456 - Deep learning](http://kurser.dtu.dk/course/02456) at DTU - Technical University of Denmark.

## Our agent

Our agent always starts in the left upper corner.


#### Playing against 3 random agents


![vs-3-random](https://github.com/eugene/pommerman/blob/master/gif/vs-3-random.gif "vs-3-random")

#### Playing against 1 simple and 2 random agents


![1simple1simple2random](https://github.com/eugene/pommerman/blob/master/gif/vs-simple-2-random.gif "Playing against 1 simple and 2 randoms")
![not_perfect](https://github.com/eugene/pommerman/blob/master/gif/not_perfect.gif "Not perfect run")


#### Playing against 3 simple agents

![1simple3simple](https://github.com/eugene/pommerman/blob/master/gif/ezgif.com-resize.gif "Playing against 3 simple")



## :wrench: Requirements
In addition to `pytorch` (https://pytorch.org) and the usual data science packages (`numpy`, `matplotlib`) this project depends on the Pommerman `playground` (https://github.com/MultiAgentLearning/playground) to be installed in your Python environment. Additionally, A2C script depends on the `colorama` package which helps with the rendering of the game in terminal (perfect for running on remote servers).

## :exclamation: DISCLAIMER
As this project uses the torch.multiprocessing package which is not compatible with jupyter notebook, the files to reproduce our results are suplied as individual python files. The guide for running these files is written below.

## :chart_with_upwards_trend: Imitation learning
To start the imitation learnining, first place the ```log_simpleAgents_sequence_observe.py``` file in the path ```playground\pommerman\cli```. Hereafter, place the file ```AA_RUN_LOG_SCRIPT.py``` in the playground folder and run it.

Observations from 10.000 games will now be collected and logged to three files in the pommerman folder. Once logging is complete, run the ```train_rnn_cnn.py``` to generate the trained imitation model.

Once the actor has been trained, the critic must also observe some games in order to learn to reward correctly before being allowed to affect the model. We do this by placing ```A3C_v10_cnn_lstm_train-critic.py``` and ```sharedAdam.py``` in the playground folder and running the the ```A3C_v10_cnn_lstm_train-critic.py``` file.

## :chart_with_upwards_trend: A3C Model
To train the A3C model, place the ```A3C_v10_cnn_lstm.py``` and ```sharedAdam``` files in the pommerman folder and run the file ```A3C_v10_cnn_lstm.py```. Inside the file you can specify a filename which will be used to save the checkpoint once the model has trained. This will also be used to load the checkpoint again if you wish to train further. The parameter MAX_EP specifies how many episodes to run before saving the checkpoint and terminating.

## :chart_with_downwards_trend: A2C Model
To generate the `convrnn-s.weights` weights file (refreshed every 300 episodes):

```python A2C/main.py train```

To see how your agent plays (loads the `convrnn-s.weights` weights file and can be used while the training is running):

```python A2C/main.py eval```

During the training current `gamma`, `running reward`, `action statistics` and `loss` are printed after each episode. It takes around 48 hours to fully train this model (40000 episodes) on a modern 10 core CPU with a single 1080TI GPU. Additionally, a `training.txt` file is generated with the main statistics for each trained episode.

## :hammer: Model
The full model that is used for this project can be seen in the below image

![fullmodel](https://github.com/eugene/pommerman/blob/master/img/architecture.PNG "The full model")

## :bar_chart: Main results
From the following figure, we see that with 40.000 episodes that A2C performas better than A3C

![a2ca3c](https://github.com/eugene/pommerman/blob/master/img/A2C_vs_A3C.png "Performance between A3C and A2c")

Finally we have the reward for our architecture shown below

![final_training](https://github.com/eugene/pommerman/blob/master/img/final_training_reward.png "Finally the final model with its reward")

## :page_facing_up: Paper
See [our paper](https://github.com/eugene/pommerman/blob/master/paper.pdf) for detailed information about the project.

## :bust_in_silhouette: Credits
* Kris Walther (s172990, [@KrisWalther](https://github.com/KrisWalther))
* Mirza Hasanbasic (s172987, [@kazyka](https://github.com/kazyka))
* Søren Hjøllund Jensen (s123669, [@SorenJ89](https://github.com/SorenJ89))
* Yevgen Zainchkovskyy (s062870, [@eugene](https://github.com/eugene))

+ Special thanks to [@dimatter](https://github.com/dimatter) for the provided computational ressources :heart:
