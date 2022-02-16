# League of Legends Simple Victory Predictor

* By Mihailo Pantić

Using Cassiopeia and scikit-learn python libraries, attempts to predict outcomes of League games, or the chance of winning one in the first place.
Sadly, due to issues with Cassiopeia, currently unable to operate on live games and states of games in specific points in time.

## Available classes:

### LOL()

#### Available functions:
* update_team()
	> Updates class variable that determines which side's players of the match will be datamined. 
	Available values: "BLUE" , "RED"
	**WARNING: User input needs to be exactly the same as one of the available values!**
	Automatically ran upon initializing an object of this function's class.
* update_region()
	> Updates class variable that determines from which region will data be obtained.
	Available values: "EUNE" , "NA" , "BR" , "EUW" , "JP" , "KR" , "LAN" , "LAS" , "OCE" , "RU" , "TR" 
	**WARNING: User input needs to be exactly the same as one of the available values!**
	Automatically ran upon initializing an object of this function's class.
* update_league()
	> Updates class variable that determines from which league will data be obtained. 
	Available values: "master" , "challenger"
	**WARNING: User input needs to be exactly the same as one of the available values!**
	Automatically ran upon initializing an object of this function's class.
* update_summoner()	
	> Updates class variable that determines a summoner's name from which will data be obtained. 
	Can be anything, but in order to get data, should be a valid summoner name.
* write_league_data()
	> Generates a .csv file containing up to 20 games from each league player, as determined from update_league().
	Data drawn consists of each summoner on the chosen team's KDA, champion, vision score and gold, as well as the game duration and whether the team won or not.
	Ignores games played by multiple people in the same league.
	Amount of players dependant on league, usually about 500 for master, 200 for challenger.
	**WARNING: Takes a long time! (dependant on number of players, and slightly on your own computer's strength)**
* get_match_ids()
	> Returns up to last 20 games of queue type RANKED SOLO of a summoner.
	Must run update_summoner() beforehand in order to work.
	Displays that summoner's champion and score of that game, as well as the match's internal id.
* get_match_by_id(match_id)
	> Updates class variable with a League match whose id is equal to this function's argument.
	Advice: Use get_match_ids() for fresh match ids.
* write_single_data()
	> Generates a .csv file containing one match worth of data.
	Data drawn consists of each summoner on the chosen team's KDA, champion, vision score and gold, as well as the game duration and whether the team won or not.
	
### Model(path_to_data)
> path_to_data is a string containing path of a .csv file with league matches relative to the script's location. 
> Uses logistic regression.
#### Available functions
* load_train_test()
	> Defines x_train, y_train, x_test and y_test within the class to be used by other functions.
	Source is the object initialization argument.
* fit()
	> Fits the model using x_train and y_train.
	 Requires x_train and y_train to be defined. Run load_train_test() before this.
* predict()
	> Generates predictions as a class variable y_pred.
	Requires x_test to be defined. Run load_train_test() before this.
* score()
	> Prints the score of the model.
		Requires x_train and y_train to be defined. Run load_train_test() before this.
* predict_single(path_to_single)
	> Prints either a prediction of a win or loss, whichever is more likely, and the prediction's certainty.
	path_to_single is a string containing path of a .csv file with one match relative to the script's location. 
	Run fit() before this.
