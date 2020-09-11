'''
WHY?

What is the most import statistic that pivots the winning/ losing of the match?
At what point is the fait of the match clear?
What are the critical values for each 'vital' statistic?


TODO:
- plot movements of the players around the court.
- measure total distance travelled.
- Animate plots such as player movement and shot distribution?
- Is Nadal being moved around the court more than Djokovic? Because Djokovic is controlling the match?
- Collate the data into a single dataframe, then I can use that to control and filter all the visualisations etc. - Not practical

'''

## Import Packages
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter as tk
import numpy as np


## Read in the data
events_data = pd.read_csv('data/events.csv',index_col=0)
points_data = pd.read_csv('data/points.csv',index_col=0)
rallies_data = pd.read_csv('data/rallies.csv',index_col=0)
# serves_data = pd.read_csv('data/serves.csv',index_col=0)

## Data organising
# events_data.reset_index(inplace=True,drop=True)
points_data.reset_index(inplace=True,drop=True)
rallies_data.set_index('rallyid',inplace=True,drop=True)
# serves_data.reset_index(inplace=True,drop=True)


class Match:
    ## This class contains everything specific to a tennis match.
    
    def __init__(self):
        self.WINNER = points_data['winner'].iloc[-1]        ## Defines the match winner as the person who wins the last point of the match.
        self.PLAYERS = list(set(points_data['server']))     ## Defines the players by taking a 'set' of the 'server' column in points_data.
        if not self.PLAYERS[0] == self.WINNER:
            self.PLAYERS = [self.WINNER,self.PLAYERS[0]]

        self.FINAL_SCORE = self.get_final_score()    ## As dict... - Taken from the score after the last point of the match. - {Sets:'3:0',Games:'6:3, 6:2, 6:3'} Winner first

        print('Players:',self.PLAYERS)
        print('Winner:',self.WINNER)
        print('Final Score:',self.FINAL_SCORE)

        self.match_stats()

    def get_final_score(self):
        '''May not support tie-break sets.'''
        score_string = points_data['score'].iloc[-1]    ## format: "6:3 6:2 6:3, 0:0"

        games_string = score_string.split(',')[0].replace(' ',', ')
        winner_sets = 0
        loser_sets = 0
        for set_ in games_string.split(', '):
            if int(set_[0]) > int(set_[set_.find(':')+1]):
                winner_sets += 1
            elif int(set_[set_.find(':')+1]) > int(set_[0]):
                loser_sets += 1
        
        sets_string = '{}:{}'.format(winner_sets,loser_sets)
        return {'Sets':sets_string,'Games':games_string}

    def match_stats(self):
        '''
        - points won
        - aces
        - faults
        - winner -> forehand / backhand
        - first serve percentage
        '''
        ## Initiate stat dicts
        self.player1 = {'name':self.PLAYERS[0],'points_won':0,'aces':0,'faults':0,'winners':{'forehand':0,'backhand':0}}
        self.points_won = {}
        self.aces = {}
        self.faults = {}
        # self.winners = {'Djokovic':{'forehand':0,'backhand':0},'Nadal':{'forehand':0,'backhand':0}}
        self.winners = {}
        self.serves = {}

        self.p1_pointsagg_ts = pd.DataFrame([{'timestamp':0,'points agg':0}])
        self.p2_pointsagg_ts = pd.DataFrame([{'timestamp':0,'points agg':0}])

        for point_index in points_data.index:
            # Clean Data
            returner = points_data['returner'].iloc[point_index]
            if not returner in self.PLAYERS or returner == '':
                points_data.at[point_index,'returner'] = list(set(self.PLAYERS).difference({points_data['server'].iloc[point_index]}))[0]

            ## Define "rallyid" (taken from the column) this allows reference to events_data.
            rallyid = int(points_data['rallyid'].iloc[point_index])
            
            # Define the "hitter" (player who hit the last shot of the point)
            hitter = events_data['hitter'].loc[events_data['rallyid']==rallyid].values[-1]
            # if int(points_data['strokes'].iloc[point_index]) % 2 == 0:
            #         hitter = points_data['returner'].iloc[point_index]
            # else:
            #     hitter = points_data['server'].iloc[point_index]

            ## Define "winner" of the point
            winner = points_data['winner'].iloc[point_index]

            ## Define "server"
            server = points_data['server'].iloc[point_index]

            ## Define "serve" ('first'/'second')
            serve = points_data['serve'].iloc[point_index]

            ## Define the "reason" for the point ending (winner, net, out, ace)
            reason = points_data['reason'].iloc[point_index]

            ## Define "hand" (the stroke type)
            hand = events_data.loc[events_data['rallyid']==rallyid].iloc[-1]['stroke']

            ##------------- STATS ---------------
            ## Points Won
            try:
                self.points_won[winner] += 1
            except KeyError:
                self.points_won[winner] = 1

            timestamp = float(events_data.loc[events_data['rallyid']==rallyid].iloc[-1]['time'])/60    # timestamp in minutes
            if winner == self.PLAYERS[0]:
                self.p1_pointsagg_ts = self.p1_pointsagg_ts.append({'timestamp':timestamp,'points agg':self.p1_pointsagg_ts['points agg'].iloc[-1]+1},ignore_index=True)
            elif winner == self.PLAYERS[1]:
                self.p2_pointsagg_ts = self.p2_pointsagg_ts.append({'timestamp':timestamp,'points agg':self.p2_pointsagg_ts['points agg'].iloc[-1]+1},ignore_index=True)
            
            if reason == 'ace':     ## ACES
                try:
                    self.aces[server][serve] += 1
                except KeyError:
                    try:
                        self.aces[server][serve] = 1
                    except KeyError:
                        self.aces[server] = {'first':0,'second':0}
                        self.aces[server][serve] += 1
            elif reason in ['out','net']:       ## FAULTS
                try:
                    self.faults[hitter][hand] += 1
                except KeyError:
                    try:
                        self.faults[hitter][hand] = 1
                    except KeyError:
                        self.faults[hitter] = {hand:1}
            elif reason == 'winner':        ## WINNERS
                if hand == '__undefined__' and events_data.loc[events_data['rallyid']==rallyid].iloc[-1]['type'] == 'smash':
                    hand = 'smash'
                try:
                    self.winners[winner][hand] += 1
                except KeyError:
                    try:
                        self.winners[winner][hand] = 1
                    except KeyError:
                        self.winners[winner] = {hand:1}
            
            ## Serve Success Tally
            try:
                self.serves[server]
            except KeyError:
                self.serves[server] = {'first':{'win':0,'lose':0,'perc':0},'second':{'win':0,'lose':0,'perc':0}}
            if winner == server:
                self.serves[server][serve]['win'] += 1
            else:
                self.serves[server][serve]['lose'] += 1

        ## Serve Percentage
        for player in self.PLAYERS:
            self.serves[player]['first']['perc'] = int(round(100 * self.serves[player]['first']['win'] /(self.serves[player]['first']['win'] + self.serves[player]['first']['lose'])))
            self.serves[player]['second']['perc'] = int(round(100 * self.serves[player]['second']['win'] /(self.serves[player]['second']['win'] + self.serves[player]['second']['lose'])))

        ## Average strokes per point
        self.average_strokes = int(round(points_data['strokes'].mean()))
        print('Average strokes:',self.average_strokes)


        print('Points Won:',self.points_won)
        print('Aces:',self.aces)
        print('Faults:',self.faults)
        print('Winners:',self.winners)
        print('Serve Percentage:',self.serves)




class Dashboard():
    ## This class contains everything specific to a dashboard
    '''
    Standard Layout is as follows...

    |---------------------------------------------|
    |                     |                       |
    |                     |                       |
    |                     |  Sub visualisations   |
    |     Court Vis       | _____________________ |
    |                     |      Key Stats        |
    |                     |                       |
    |                     |                       |
    |                     |                       |
    |---------------------------------------------|

    The Sub Visualisations would be in a grid layout, accommodating for various numbers of 'sub-vises'.
    The Court Vis would show a diagram of the court, with interesting data plotted on top - such as shot distribution.

    Key stats:
    - Aces
    - Double faults
    - 1st Serve %
    - 1st Serve points won
    - 2nd Serve points won
    - Winners
    - Unforced Errors
    - Net Points Won
    - Break points won x/X
    '''

    LARGE_FONT = ('Verdana',16)
    MEDIUM_FONT = ('Verdana',14)
    SMALL_FONT = ('Verdana',12)

    def __init__(self):
        self.fixed_size = False
        self.w_width = 1300
        self.w_height = 700

        self.ppi = 100

        self.subvises = []
        self.court_fig = None

        self.root = tk.Tk()
        self.root.wm_title('Tennis Analysis')
        self.root.wm_geometry('{}x{}'.format(self.w_width,self.w_height))
        if self.fixed_size:
            self.root.resizable(False,False)

        # self.root = tk.Tk()
        # self.window = tk.Frame(self.root)
        # self.window.pack(side='top',fill='both',expand=True)

        title = tk.Label(self.root,text='Dashboard',font=self.LARGE_FONT)
        # title.grid_configure(row=0,column=0,columnspan=2)
        title.pack(side=tk.TOP,fill=tk.BOTH,expand=True,padx=10,pady=10)

        self.court_vis = tk.Frame(self.root,width=self.w_width/2,height=self.w_height/2)
        # self.court_vis.grid(row=1,column=0,rowspan=2)
        self.court_vis.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=10,pady=10)

        self.subvis = tk.Frame(self.root,width=self.w_width/2,height=self.w_height/2)
        # self.subvis.grid(row=1,column=1,stick='nsew')
        self.subvis.pack(side=tk.LEFT,fill=tk.BOTH,expand=False,padx=10,pady=10)


        
    def _quit(self):
        print('quit')
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def show_window(self):
        # Overriding the close window (using x button) because it wasnt properly breaking from the 'mainloop'
        def on_closing():
            print('closing')
            self.root.quit()
            self.root.destroy()
            print('destroyed')
        self.root.protocol("WM_DELETE_WINDOW",on_closing)
        self.root.mainloop()
    
    def add_visualisations(self):
        # Adding a MPL figure to the window.
        if not self.court_fig is None:
            canvas = FigureCanvasTkAgg(self.court_fig,master=self.court_vis)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        else:
            print('Court figure is not defined.')

        if len(self.subvises) > 0:
            if len(self.subvises) == 1:
                fig = self.subvises[0]
                if self.fixed_size:
                    fig.set_size_inches((self.w_width)/self.ppi,(self.w_height/2)/self.ppi)   # Size in inches
                    fig.tight_layout()
                canvas = FigureCanvasTkAgg(fig,master=self.subvis)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            else:
                for index, fig in enumerate(self.subvises):
                    
                    if self.fixed_size:
                        if len(self.subvises) % 2 != 0 and index == len(self.subvises) - 1:
                            width = (self.w_width/2)/self.ppi
                        else:
                            width = (self.w_width/4)/self.ppi
                        height = (self.w_height/2)/self.ppi
                        fig.set_size_inches(width,height)
                        fig.tight_layout()
                    else:
                        oldSize = fig.get_size_inches()
                        factor = 0.75
                        fig.set_size_inches([factor * s for s in oldSize])
                    column = index % 2
                    row = int((index / 2) % 2)
                    canvas = FigureCanvasTkAgg(fig,master=self.subvis)
                    canvas.draw()
                    if len(self.subvises) % 2 != 0 and index == len(self.subvises) - 1:
                        canvas.get_tk_widget().grid(row=row,column=column,columnspan=2,stick='NSEW')
                    else:
                        canvas.get_tk_widget().grid(row=row,column=column,stick='NSEW')
        else:
            print('There are no sub visualisations available to add to dashboard.')

    def tornado_plot(self,values,labels=None,thickness=0.5,separation=0.1,show=False):
        '''
        Create a tornado plot using a simple mpl plot and blank figure and indivually
        placing each line to form the 'boxes'.

        values: list/ tuple of tuples containing the left and right values for each category
        labels: list of strings describing each category
        '''
        ## Take base as half and normalise all values.
        base = 0.5
        values_ = []
        max_ = max(max(values))
        for tup in values:
            values_.append(((tup[0]/max_)*base,(tup[1]/max_)*base))
        values = values_

        ## Draw
        fig, ax = plt.subplots()
        ys = []
        for i, set_ in enumerate(values):
            y = i * (thickness + separation * thickness)
            ys.append(y+thickness/2)
            ax.broken_barh([(base-set_[0],set_[0]),(base,set_[1])],(y,thickness),facecolors=['tab:orange','tab:blue']) # TODO: consistent colours for each player.

        ax.set_xlim(base-max(max(values))-0.05,base+max(max(values))+0.05)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.title('Stats Bars')
        plt.xticks([])
        plt.yticks(ys,labels)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        plt.tight_layout()
        if show:
            plt.show()
        
        self.subvises.append(fig)


    def pie_chart(self,data,title=None,labels=None):       # TODO: labels are cut off
        # pie_fig = plt.figure(figsize=(4,4),dpi=100)
        # # pie_fig = plt.figure()
        # pie_axes = pie_fig.add_axes([0.1,0.1,0.8,0.8])
        fig, axes = plt.subplots()
        explode = []
        for _ in range(len(data)):
            explode.append(0.01)
        print(explode)
        axes.pie(data,labels=labels,autopct='%1i%%',startangle=90,counterclock=False,explode=explode)
        axes.axis('equal')
        if title is not None:
            plt.title(title)
        self.subvises.append(fig)




class Tennis_Analysis:
    ## This is the high level class that contains everything else that is used in this programme.
    def __init__(self,dashboard=False):
        self.match = Match()

        if dashboard:
            self.dash = Dashboard()

            self.shot_distribution()

            self.stats_tornado()

            # pie_data = [self.match.points_won[self.match.PLAYERS[0]],self.match.points_won[self.match.PLAYERS[1]]]
            # self.dash.pie_chart(pie_data,title='Points Won',labels=self.match.PLAYERS)
            
            self.points_time_series()


            self.dash.add_visualisations()
            self.dash.show_window()
        else:
            self.dashboard = False

    def combine_data_sources(self): ## TODO
        """
        Assuming I know the names of the sources.
        Going to form large point by point data - adding rally info etc.
        """

        # comb_data = pd.DataFrame(columns=['PointID','RallyID','ServeID'])
        pass

    def move_figure(self,fig,x,y):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = mpl.get_backend()
        if backend == 'TkAgg':
            fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        else:
            print('Invalid Backend [{}]'.format(backend))


    def plot_court(self):
        ## Initiate the figure and axes
        self.court_fig = plt.figure(figsize=(3,6),dpi=100)
        self.court_fig.tight_layout()
        self.court_axes = self.court_fig.add_axes([0.1,0.1,0.8,0.8],facecolor='#43508D')

        # Co-ords for all court lines except 'net' and 'Ts'
        FullCourt_X = [0,0,10.973,10.973,0,1.372,1.372,9.601,9.601,1.372,1.372,1.372,9.601,9.601,1.372,5.487,5.487]
        FullCourt_Y = [0,23.774,23.774,0,0,0,23.774,23.774,0,0,5.486,18.288,18.288,5.486,5.486,5.486,18.288]

        self.court_axes.plot(FullCourt_X,FullCourt_Y,color='white',linewidth=1)

        # Net
        X = [0,10.973]
        Y = [11.887,11.887]
        self.court_axes.plot(X,Y,color='white',linewidth=1)

        # Baseline Ts
        X = [5.487,5.487]
        Y = [0,0.15]
        self.court_axes.plot(X,Y,color='white',linewidth=1)
        X = [5.487,5.487]
        Y = [23.774,23.624]
        self.court_axes.plot(X,Y,color='white',linewidth=1)

        # Add space around the court.
        self.court_axes.set_xlim([-2,13])
        self.court_axes.set_ylim([-6,30])

    def shot_distribution(self):
        ## Attempt to show the last shot locations (from points_data) on a diagram of the court
        self.plot_court()
        for point_index in range(len(points_data.index)):
            if points_data['reason'].iloc[point_index] == 'winner':
                # if int(points_data['strokes'].iloc[point_index]) % 2 == 0:
                #     hitter = points_data['returner'].iloc[point_index]
                # else:
                #     hitter = points_data['server'].iloc[point_index]
                rallyid = int(points_data['rallyid'].iloc[point_index])
                hitter = events_data['hitter'].loc[events_data['rallyid']==rallyid].values[-1]
                x = float(points_data['x'].iloc[point_index])
                y = float(points_data['y'].iloc[point_index])
                if hitter == self.match.PLAYERS[0]:
                    if points_data['winner'].iloc[point_index] != hitter:
                        self.court_axes.plot(x,y,'r+',markersize=3)
                    else:
                        self.court_axes.plot(x,y,'y+',markersize=3)
                elif hitter == self.match.PLAYERS[1]:
                    if points_data['winner'].iloc[point_index] != hitter:
                        self.court_axes.plot(x,y,'rx',markersize=3)
                        print(points_data.iloc[point_index])
                    else:
                        self.court_axes.plot(x,y,'yx',markersize=3)        
        plt.title('Shot Distribution')
        
        if self.dashboard:
            self.dash.court_fig = self.court_fig
        else:
            plt.show()


    def stats_label(self,data=None):
        # Data passed in the form of an array of each statistical data pair.
        '''
        The aim is to have 'centralised' bar charts to indicate the statistics in a aesthetic way. -- DONE - use stats_tornado() (wraps tornado_plot())
        '''
        stats_string = '''
        Aces: {}\n
        Points Won: {}\n
        Winners: {}\n
        Faults: {}\n
        '''.format(ta.match.aces,ta.match.points_won,ta.match.winners,ta.match.faults)

        self.dash.stats_container = tk.Frame(self.dash.root)
        self.dash.stats_container.grid(row=2,column=1,stick='nsew')
        self.dash.head = tk.Label(self.dash.stats_container,text='Stats Label',font=self.dash.MEDIUM_FONT)
        self.dash.head.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

        self.dash.info = tk.Label(self.dash.stats_container,text=stats_string,font=self.dash.SMALL_FONT)
        self.dash.info.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

    def stats_tornado(self):
        labels = ['Aces','Points Won','Winners','Faults']
        values = [
            (sum(self.match.aces[self.match.PLAYERS[0]].values()),sum(self.match.aces[self.match.PLAYERS[1]].values())),
            (self.match.points_won[self.match.PLAYERS[0]],self.match.points_won[self.match.PLAYERS[1]]),
            (sum(self.match.winners[self.match.PLAYERS[0]].values()),sum(self.match.winners[self.match.PLAYERS[0]].values())),
            (sum(self.match.faults[self.match.PLAYERS[0]].values()),sum(self.match.faults[self.match.PLAYERS[0]].values()))
        ]

        self.dash.tornado_plot(values,labels)
        

    def points_time_series(self,animate=False):
        # fig = plt.figure(figsize=(4,4),dpi=100)
        fig, ax = plt.subplots()

        # print(self.p1_timeseries)
        # print(self.p2_timeseries)
        ax.plot(self.match.p1_pointsagg_ts['timestamp'].values,self.match.p1_pointsagg_ts.index,label=self.match.PLAYERS[0])
        ax.plot(self.match.p2_pointsagg_ts['timestamp'].values,self.match.p2_pointsagg_ts.index,label=self.match.PLAYERS[1])

        ## TODO: Add annotations on the plot to divide each set.

        plt.xlabel('Timestamp (Mins)')
        plt.ylabel('Points Aggregate')
        plt.title('Points Progression by Time')
        plt.legend()

        plt.grid()
        # plt.show()
        self.dash.subvises.append(fig)





##### RUNNING 

ta = Tennis_Analysis(dashboard=False)

    

