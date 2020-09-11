"""
--------Machine Learning Application - Tennis Analysis---------


Use the point information to make judgements about how Nadal should have played his shots to win the match - 'Win the match' defined as just winning more points.


Information to be used:
- Server
- First/ second
- Serve location
- Returner location
- Returner shot type

Model Info:
- binary classifier
- Options
    - Decision Tree/ RF
    - Logistic Regression

Assumptions:
- "__undefined__" smash is a forehand
- "return" shot type is taken to be its own type when in reality it is a 'block' or 'slice' - shouldn't matter what it actually represents (unless it is a slice in which case that category is losing potential frequency.)

Adhoc dependencies:
- Tennis_Analysis.py
"""

## Imports
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from Tennis_Analysis import Tennis_Analysis as TA
TA = TA()

## Data injestion
events_data = pd.read_csv('data/events.csv',index_col=0)    # 865 recs
points_data = pd.read_csv('data/points.csv',index_col=0)    # 142 recs
rallies_data = pd.read_csv('data/rallies.csv',index_col=0)  # 206 recs
serves_data = pd.read_csv('data/serves.csv',index_col=0)    # 133 recs

## Clean data
print('Cleaning data...')
for index in range(len(events_data.index)):
    event = events_data.iloc[index]
    if event['receiver'] not in TA.match.PLAYERS or event['receiver'] == '__undefined__':
        events_data['receiver'].iloc[index] = set(TA.match.PLAYERS).difference(event['hitter'])


points_data.reset_index(inplace=True,drop=True)
for index in range(len(points_data.index)):
    point = points_data.iloc[index]
    if point['returner'] not in TA.match.PLAYERS or point['returner'] == '__undefined__':
        points_data['returner'].iloc[index] = set(TA.match.PLAYERS).difference(point['server'])




## Combine the data to 1 df (where possible)
if False:
    comb_data = pd.DataFrame(columns=['pointid','server','serve','serve_x','serve_y','server_shot_type','server_shot_hand','server_x','server_y','returner_x','returner_y','returner_shot_type','returner_shot_hand','num_strokes','is_game_point','is_break_point','winner'])

    bp_next = False # break point
    gp_next = False  # game point
    for index in points_data.index:
        pointid = index
        server = points_data['server'].iloc[index]
        returner = points_data['returner'].iloc[index]
        serve = points_data['serve'].iloc[index]
        if points_data['rallyid'].iloc[index] not in serves_data['rallyid'].values:
            continue        ## If the rallyid is not available - skip record.
        serve_x = float(serves_data['x'].loc[serves_data['rallyid']==points_data['rallyid'].iloc[index]].values[0])
        serve_y = float(serves_data['y'].loc[serves_data['rallyid']==points_data['rallyid'].iloc[index]].values[0])

        server_shot_type, server_shot_hand, server_x, server_y, returner_x, returner_y, returner_shot_type, returner_shot_hand = None, None, None, None, None, None, None, None
        for i in events_data.loc[events_data['rallyid']==points_data['rallyid'].iloc[index]].index:
            if events_data['hitter'].iloc[i] == server:
                server_shot_type = events_data['type'].iloc[i]
                server_shot_hand = events_data['stroke'].iloc[i]
                if server_shot_hand == '__undefined__':
                        server_shot_hand = 'forehand'
                server_x = float(events_data['hitter_x'].iloc[i])
                server_y = float(events_data['hitter_y'].iloc[i])
            else:
                returner_x = float(events_data['hitter_x'].iloc[i])
                returner_y = float(events_data['hitter_y'].iloc[i])
                returner_shot_type = events_data['type'].iloc[i]
                returner_shot_hand = events_data['stroke'].iloc[i]
                if returner_shot_hand == '__undefined__':
                        returner_shot_hand = 'forehand'

        num_strokes = points_data['strokes'].iloc[index]
        winner = points_data['winner'].iloc[index]

        is_game_point, is_break_point = 0,0
        _game_score = points_data['score'].iloc[index].split(' ')[-1]
        if _game_score.split(':')[0] == '40' and _game_score.split(':')[1] != '40':
            if server == TA.match.PLAYERS[0]:
                gp_next = True
            else:
                bp_next = True
        elif _game_score.split(':')[0] == 'Ad':
            if server == TA.match.PLAYERS[0]:
                gp_next = True
            else:
                bp_next = True
        elif _game_score.split(':')[1] == '40' and _game_score.split(':')[0] != '40':
            if server == TA.match.PLAYERS[1]:
                gp_next = True
            else:
                bp_next = True
        elif _game_score.split(':')[1] == 'Ad':
            if server == TA.match.PLAYERS[1]:
                gp_next = True
            else:
                bp_next = True
        elif _game_score == '0:0':
            if gp_next:
                is_game_point = 1
                is_break_point = 0
            elif bp_next:
                is_game_point = 0
                is_break_point = 1
        else:
            gp_next, bp_next = False, False


        val_dict = {'pointid':pointid,'server':server,'serve':serve,'serve_x':serve_x,'serve_y':serve_y,'server_shot_type':server_shot_type,'server_shot_hand':server_shot_hand,'server_x':server_x,'server_y':server_y,'returner_x':returner_x,'returner_y':returner_y,'returner_shot_type':returner_shot_type,'returner_shot_hand':returner_shot_hand,'num_strokes':num_strokes,'is_game_point':is_game_point,'is_break_point':is_break_point,'winner':winner}

        comb_data = comb_data.append(val_dict,ignore_index=True)

    comb_data.dropna(axis=0,inplace=True)

    comb_data.to_csv('combined_data.csv')
else:
    comb_data = pd.read_csv('combined_data.csv',index_col=0)
    pass


## Convert to numerical keys
comb_data.drop(['pointid'],axis=1,inplace=True)

players_key_dict = {
        TA.match.PLAYERS[0]:0,
        TA.match.PLAYERS[1]:1
}

comb_data['server'] = comb_data['server'].map(players_key_dict)
comb_data['serve'] = comb_data['serve'].map({'first':0,'second':1})

shot_key_dict = {}
for i, shot in enumerate(list(set(comb_data['server_shot_type'].values))):
    shot_key_dict[shot] = i
comb_data['server_shot_type'] = comb_data['server_shot_type'].map(shot_key_dict)

hand_key_dict = {'forehand':0,'backhand':1}
comb_data['server_shot_hand'] = comb_data['server_shot_hand'].map(hand_key_dict)

for i, shot in enumerate(list(set(comb_data['returner_shot_type'].values))):
    try:
        shot_key_dict[shot]
    except:
        shot_key_dict[shot] = len(shot_key_dict)
comb_data['returner_shot_type'] = comb_data['returner_shot_type'].map(shot_key_dict)
comb_data['returner_shot_hand'] = comb_data['returner_shot_hand'].map(hand_key_dict)

comb_data['winner'] = comb_data['winner'].map(players_key_dict)


comb_data.to_csv('comb_data_numerical.csv')

## Normalise Numbers - otherwise importance weightings were not intuitive.
data = comb_data.values
normalised_data = preprocessing.MinMaxScaler().fit_transform(data)
normalised_df = pd.DataFrame(normalised_data,columns=comb_data.columns)
normalised_df.to_csv('normalised_data.csv')


X = normalised_df.drop(['winner'],axis=1).values
y = normalised_df['winner'].values

## ML Model (15 features)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10,random_state=0)


models = {
    'Logistic Regression':LogisticRegression(multi_class='auto',solver='liblinear',random_state=0),
    'Decision Tree Classifier':DecisionTreeClassifier(criterion='gini',random_state=0),
    'Decision Tree Classifier (entropy)':DecisionTreeClassifier(criterion='entropy',random_state=0),
    'Decision Tree Regressor':DecisionTreeRegressor(random_state=0),
    'Random Forest Classifier':RandomForestClassifier(criterion='gini',random_state=0,n_estimators=100),
    'Neural Network':tf.keras.models.Sequential([
        tf.keras.layers.Dense(15,activation=tf.nn.relu),
        tf.keras.layers.Dense(2,activation=tf.nn.softmax)
    ])
}

best_model = (None,0,None)  ## (Name , test_acc, model)
for model_name in list(models.keys()):
    print('Model:',model_name)
    model = models[model_name]

    if model_name == 'Neural Network':
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy']) ## TODO: Improve nn.
        model.fit(X_train,y_train,epochs=55)

        test_loss, test_acc = model.evaluate(X_test, y_test)

        print('Test accuracy:', test_acc)
    else:
        model.fit(X_train,y_train)
        prediction = model.predict(X_test)

        correct = 0
        for i in range(len(prediction)):
            if prediction[i] == y_test[i]:
                correct += 1

        test_acc = correct/len(prediction)
        perc = round(100*test_acc,1)
        print('{}/{} correct ({}%)'.format(correct,len(prediction),perc))
    
    if test_acc > best_model[1]:
        best_model = (model_name , test_acc, model)

print('Best model is: {} [ {}% ]'.format(best_model[0],round(100*best_model[1],1)))

print(np.std(X_train,0)*best_model[2].coef_)

feat_importance = np.std(X_train,0)*best_model[2].coef_[0]
for i, x in enumerate(feat_importance):
    print(list(comb_data.columns)[i],x)
