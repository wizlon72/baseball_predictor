# baseball_predictor
python scrip to build logistic regression model and predict results of upcoming games

Usage: python buildregression.py - to build model and compare model against today's games
Usage: python webscraper.py - can be used independently to scrape information about todays games and probable starters

Note - all historical data needs to be manually updated - model is currently configured to build against 2017 data analyzing impact ERA difference and Day/Night has on home team win chance. 
Both the historical game logs and pitcher ERA data needs to be manually updated.
To evaluate the model with team performance edits need to be made to include the dummy variable for all team names in the model building portion as well as corresponding edits to add these variables in the test sets.
