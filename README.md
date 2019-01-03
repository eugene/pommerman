# Pommerman :bomb:
PyTorch based, reinforcement learning solution for the [Pommerman competitions](https://www.pommerman.com/) done as an exam project in course [02456 - Deep learning](http://kurser.dtu.dk/course/02456) at DTU - Technical University of Denmark.

## :wrench: Requirements 
In addition to `pytorch` (https://pytorch.org) and the usual data science packages (`numpy`, `matplotlib`) this project depends on the Pommerman `playground` (https://github.com/MultiAgentLearning/playground) to be installed in your Python environment. Additionally, A2C script depends on the `colorama` package which helps with the rendering of the game in terminal (perfect for running on remote servers).

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

## :bar_chart: Main results
Business model canvas interaction design partner network pitch business plan. Business-to-consumer accelerator startup partnership validation ramen network effects branding metrics monetization mass market. Beta bandwidth twitter facebook seed money innovator vesting period social proof pivot.

## :page_facing_up: Paper
Business model canvas interaction design partner network pitch business plan. Business-to-consumer accelerator startup partnership validation ramen network effects branding metrics monetization mass market. Beta bandwidth twitter facebook seed money innovator vesting period social proof pivot.

## :bust_in_silhouette: Credits
* Kris Walther (s172990, https://github.com/KrisWalther)
* Mirza Hasanbasic (s172987, https://github.com/kazyka)
* SÃ¸ren HjÃ¸llund Jensen (s123669, https://github.com/SorenJ89)
* Yevgen Zainchkovskyy (s062870, https://github.com/eugene)