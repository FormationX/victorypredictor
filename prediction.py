import cassiopeia as cass
from cassiopeia.core import Summoner, MatchHistory, Match
from cassiopeia import Queue, Patch
import time
import csv
import sklearn as sk
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

class LOL():
    parameters=["region", "league", "summoner", "team"]
    regions=["EUNE","NA","BR","EUW","JP","KR","LAN","LAS","OCE","RU","TR"]
    leagues=["master", "challenger"]
    teams=["BLUE", "RED"]
    summoner=""
    team=""
    qtype="RANKED_SOLO_5x5"
    
    def __init__(self):
        cass.set_riot_api_key("RGAPI-b9e72e71-4463-40a1-9c9e-8e380d08c9ba")  
        self.update_region()
        self.update_league()
        self.update_team()
        
    def __ask_for_input(self, question):  
        """
        Asks for user input of a question.

        Parameters
        ----------
        question : string defining what the user should enter

        Returns
        -------
        entered : user input

        """
        print("Enter " + question + " of choice:")
        entered=input()
        return entered
    
    def __print_available(self, array):
        """
        Prints data from an array to compare with user input.

        Parameters
        ----------
        array : an array

        Returns
        -------
        string : "Accepted inputs:" with array variables following with whitespace

        """
        string="Accepted inputs: "
        for data in array:
            string=string + str(data) + " "
        return string
    
    def __get_player_data(self, players):
        """
        Gets following data from each participant: KDA, Champion, Vision score, Gold, Team

        Parameters
        ----------
        players : array with elements of type Participant

        Returns
        -------
        player_data : array with element order: [KDA, Champion, Vision score, Gold, Team]

        """
        player_data=[]
        for player in players:
            player_stats=[] #[KDA, Champion, Vision score, Gold, Team]
            if player.stats.deaths==0:
                player_stats.append(((player.stats.kills +player.stats.assists )/ (player.stats.deaths+0.1)))
            else:
                player_stats.append(((player.stats.kills +player.stats.assists )/ player.stats.deaths))
            player_stats.append(player.champion.id)
            player_stats.append(player.stats.vision_score)
            player_stats.append(player.stats.gold_earned)
            if player.team.side.value==100:
                player_stats.append("BLUE")
            else:
                player_stats.append("RED")
            player_data.append(player_stats)
        return player_data
    
    def update_team(self):
        """
        Updates team class variable through user input.

        Parameters
        ----------
        none

        Returns
        -------
        none

        """
        while True:
            print(self.__print_available(self.teams))
            self.team=self.__ask_for_input(self.parameters[3])
            if self.team in self.teams:
                break
            else:
                print("Invalid " + self.parameters[3] + " ! Please enter a valid " + self.parameters[3] + ".")
                
    def update_region(self):
        """
        Updates region class variable through user input.

        Parameters
        ----------
        none

        Returns
        -------
        none

        """
        while True:
            print(self.__print_available(self.regions))
            self.region=self.__ask_for_input(self.parameters[0])
            if self.region in self.regions:
                break
            else:
                print("Invalid " + self.parameters[0] + " ! Please enter a valid " + self.parameters[0] + ".")
                
    def update_league(self):
        """
        Updates league class variable through user input.

        Parameters
        ----------
        none

        Returns
        -------
        none

        """
        while True:
            print(self.__print_available(self.leagues))
            self.league=self.__ask_for_input(self.parameters[1])
            if self.league in self.leagues:
                break
            else:
                print("Invalid " + self.parameters[1] + " ! Please enter a valid " + self.parameters[1] + ".")
    
    def update_summoner(self):
        """
        Updates summoner class variable through user input.

        Parameters
        ----------
        none

        Returns
        -------
        none

        """
        self.summoner=self.__ask_for_input(self.parameters[2])
    
    def __get_data(self):
        """
        Draws data from Riot using previously set variables.
        
        Parameters
        ----------
        None.

        Returns
        -------
        final_data : relatively badly structured array of metrics
            array of arrays: [[KDA, Champion, Vision score, Gold, Team], Baron kills, Dragon kills, Duration, Victory]

        """
        
        region=self.region
        league=self.league
        qtype=self.qtype
        observed_team=self.team
        if league=="master":
            origin=cass.get_master_league(queue=qtype, region=region)
        if league=="challenger":
            origin=cass.get_challenger_league(queue=qtype, region=region)
        match_ids=[] #duplicate comparison array
        final_data=[] #this will get converted into appropriate input for classifier model
        for entry in origin:
            match_history=entry.summoner.match_history
            for match in match_history:
                if match.queue.value==qtype and match.id not in match_ids: #ignores duplicates
                    data=[]
                    match_ids.append(match.id)
                    players=match.participants
                    del players[10:]
                    if observed_team=="BLUE":
                        player_data=self.__get_player_data(players[0:5])
                        data.append(player_data)
                        #data.append(match.teams[0].baron_kills) #these somehow dont exist?
                        #data.append(match.teams[0].dragon_kills)
                        data.append(match.duration.total_seconds())
                        data.append(int(match.teams[0].win))
                    if observed_team=="RED":
                        player_data=self.__get_player_data(players[5:10])
                        data.append(player_data)
                        #data.append(match.teams[1].baron_kills)
                        #data.append(match.teams[1].dragon_kills)
                        data.append(match.duration.total_seconds())
                        data.append(int(match.teams[1].win))
                    final_data.append(data)
        return final_data
    
    def __prepare_data(self):
        """
        Molds data obtained from __get_data into
        something usable by scikit
        
        Parameters
        ----------
        None.

        Returns
        -------
        fixed_data: 2D array

        
        """
        
        data=self.__get_data()
        fixed_data=[]
        for game in data:
            data_slice=[]
            for i in range(0,5):
                for j in range(0,4):
                    data_slice.append(game[0][i][j])
            #print(data_slice)
            for k in range(1,3):
                data_slice.append(game[k])
            fixed_data.append(data_slice)
        return fixed_data
                
    def write_league_data(self):
        """
        Creates a .csv file to be worked on with extracted data from leagues.
        WARNING: Takes a very long time! (~2 hours)

        Returns
        -------
        None.

        """
        data=self.__prepare_data()
        header=["Summoner 1 KDA" , "Summoner 1 Champion ID" , "Summoner 1 Vision score" , "Summoner 1 Gold" , 
                "Summoner 2 KDA" , "Summoner 2 Champion ID" , "Summoner 2 Vision score" , "Summoner 2 Gold" , 
                "Summoner 3 KDA" , "Summoner 3 Champion ID" , "Summoner 3 Vision score" , "Summoner 3 Gold" , 
                "Summoner 4 KDA" , "Summoner 4 Champion ID" , "Summoner 4 Vision score" , "Summoner 4 Gold" , 
                "Summoner 5 KDA" , "Summoner 5 Champion ID" , "Summoner 5 Vision score" , "Summoner 5 Gold" , 
                "Game duration" , "Victory"]
        string="train" + " " + self.region + " " + self.team + str(self.league) + " " + str(round(time.time() * 1000)) + ".csv"
        with open(string, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for game in data:
                line=[]
                for metric in game:
                    line.append(metric)
                writer.writerow(line)
                
    def get_match_ids(self):
        """
        Checks last 20 games of a summoner previously updated, and prints any games of type self.qtype

        Returns
        -------
        Prints up to 20 match ids for use with get_match_by_id

        """
        summoner=Summoner(name=self.summoner, region=self.region)
        matches=cass.get_match_history(puuid=summoner.puuid, continent="EUROPE", region=self.region, begin_index=0, end_index=200)
        #matches=summoner.match_history
        match_ids=[]
        for match in matches:
            if match.queue.value==self.qtype:
                match_ids.append(match.id)                
                players=match.participants
                del players[10:]
                for participant in players:
                    if participant.summoner.name==summoner.name:
                        print(summoner.name + " as " + participant.champion.name)
                        print(str(participant.stats.kills) + "/" + str(participant.stats.deaths) + "/" + str(participant.stats.assists))
                print("Match id = " + str(match.id))
                print()
                
    def get_match_by_id(self, match_id):
        """
        Set a match to be extracted using its id.

        Parameters
        ----------
        match_id : match id of type int, get with get_match_ids

        Returns
        -------
        cmp : Match object with chosen match id.

        """
        summoner=Summoner(name=self.summoner, region=self.region)
        matches=cass.get_match_history(puuid=summoner.puuid, continent="EUROPE", region=self.region, begin_index=0, end_index=200)
        #matches=summoner.match_history
        #print(len(matches))
        for match in matches:
            if match.id==match_id:
                cmp=match
        self.single=cmp
        
    def __prepare_single_data(self):
        """
        Works the same as __get_data and __prepare_data combined, but for a single match.

        Returns
        -------
        fixed_data : 1D array

        """
        match=self.single
        players=match.participants
        fixed_data=[]
        data=[]
        observed_team=self.team
        del players[10:]
        if observed_team=="BLUE":
            player_data=self.__get_player_data(players[0:5])
            data.append(player_data)
            #data.append(match.teams[0].baron_kills) #these somehow dont exist?
            #data.append(match.teams[0].dragon_kills)
            data.append(match.duration.total_seconds())
            data.append(int(match.teams[0].win))
        if observed_team=="RED":
            player_data=self.__get_player_data(players[5:10])
            data.append(player_data)
            #data.append(match.teams[1].baron_kills)
            #data.append(match.teams[1].dragon_kills)
            data.append(match.duration.total_seconds())
            data.append(int(match.teams[1].win))
        data_slice=[]
        for i in range(0,5):
            for j in range(0,4):
                data_slice.append(data[0][i][j])
        for k in range(1,3):
            data_slice.append(data[k])
        fixed_data.append(data_slice)
        return fixed_data
    
    def write_single_data(self):
        """
        Creates a .csv file to be worked on with extracted data from a single match.

        Returns
        -------
        None.

        """
        data=self.__prepare_single_data()
        header=["Summoner 1 KDA" , "Summoner 1 Champion ID" , "Summoner 1 Vision score" , "Summoner 1 Gold" , 
                "Summoner 2 KDA" , "Summoner 2 Champion ID" , "Summoner 2 Vision score" , "Summoner 2 Gold" , 
                "Summoner 3 KDA" , "Summoner 3 Champion ID" , "Summoner 3 Vision score" , "Summoner 3 Gold" , 
                "Summoner 4 KDA" , "Summoner 4 Champion ID" , "Summoner 4 Vision score" , "Summoner 4 Gold" , 
                "Summoner 5 KDA" , "Summoner 5 Champion ID" , "Summoner 5 Vision score" , "Summoner 5 Gold" , 
                "Game duration" , "Victory"]
        string="single" + " " + self.region + " " + self.team + self.summoner + " " + str(round(time.time() * 1000)) + ".csv"
        with open(string, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for game in data:
                line=[]
                for metric in game:
                    line.append(metric)
                writer.writerow(line)
            
class Model():
    model=LogisticRegression(max_iter=100000)
    def __init__(self, trainpath):
        self.train = pd.read_csv(trainpath)
    def load_train_test(self):
        """
        Spreads entered data into class variables for use in
            other functions.

        self.x_train
        self.y_train
        self.x_test and
        self.y_test all become available as class variables.

        """
        train, test = train_test_split(self.train, test_size=0.25)
        self.x_train=train.iloc[:,:-1]
        self.y_train=train.iloc[:,-1]
        self.x_test=test.iloc[:,:-1]
        self.y_test=test.iloc[:,-1]
    def fit(self):
        """
        Fits model based on initiated data.

        Requires x_train and y_train initiated in-class (run load_train_test() before this).
        """
        self.model.fit(self.x_train, self.y_train)
    def predict(self):
        """
        Enters prediction into a class variable.

        Requires x_test initiated in-class (run load_train_test() before this).

        self.y_pred becomes available.

        """
        self.y_pred=self.model.predict(self.x_test)
    def score(self):
        """
        Multiplied by 100, prints a percentage showing correctness.

        """
        print(self.model.score(self.x_train,self.y_train))
    def difference(self):
        """
        Not available currently.

        (prints plots)

        """
        plt.plot(list(self.y_pred))
        plt.plot(list(self.y_test))
    def predict_single(self, singlepath):
        """
        Predicts outcome of a single game.

        Parameters
        ----------
        singlepath : folder path to a .csv file containing one match worth of data.

        Prints percentage of certainty.

        """
        single=pd.read_csv(singlepath)
        x=single.iloc[:,:-1]
        y=single.iloc[:,-1]
        prediction=self.model.predict(x)
        probability=self.model.predict_proba(x)
        if prediction[0]==y[0]:
            if prediction[0]==0:
                print("Predicted a loss correctly with " + str(probability[0][0]*100) + "% certainty")
            if prediction[0]==1:
                print("Predicted a win correctly with " + str(probability[0][1]*100) + "% certainty")

                    
                    
                    
                    

                    
                    
                    
        

