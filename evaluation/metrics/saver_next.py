import pandas as pd

class Saver:
    '''
    Popularity( length=20 )

    Used to iteratively calculate the average overall popularity of an algorithm's recommendations. 

    Parameters
    -----------
    length : int
        Coverage@length
    '''
    
    session_key = 'SessionId'
    item_key    = 'ItemId'
    
    def __init__(self, length=50):
        self.length = length;
        self.sum = 0
        self.tests = 0
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        self.train_actions = len( train.index )
        #group the data by the itemIds
        grp = train.groupby(self.item_key)
        #count the occurence of every itemid in the trainingdataset
        self.pop_scores = grp.size()
        self.count = grp.size()
        #sort it according to the  score
        self.pop_scores.sort_values(ascending=False, inplace=True)
        #normalize
        self.pop_scores = self.pop_scores / self.pop_scores[:1].values[0]

    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.recommendations = {}
        self.recommendations['SessionId'] = []
        self.recommendations['Position'] = []
        self.recommendations['CurItem'] = []
        self.recommendations['NextItem'] = []
        self.recommendations['NextItem_rank'] = []
        self.recommendations['NextItem_rr'] = []
        self.recommendations[f'NextItem_{self.length}'] = []
        self.recommendations['NextItem_pop'] = []
        self.recommendations['NextItem_count'] = []
     
    def skip(self, for_item = 0, session = -1 ):
        pass 
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        #only keep the k- first predictions
        recs = result[:self.length]
        self.recommendations['SessionId'].append( session )
        self.recommendations['Position'].append( position )
        self.recommendations['CurItem'].append( for_item )
        self.recommendations['NextItem'].append( next_item )
        if next_item in recs.index:
            self.recommendations['NextItem_rank'].append( recs.index.get_loc( next_item )+1 )
            self.recommendations['NextItem_rr'].append( 1/(recs.index.get_loc( next_item )+1) )
        else:
            self.recommendations['NextItem_rank'].append( 0 )
            self.recommendations['NextItem_rr'].append( 0 )
        self.recommendations[f'NextItem_{self.length}'].append( int(next_item in recs.index) )
        self.recommendations['NextItem_pop'].append(self.pop_scores[next_item])
        self.recommendations['NextItem_count'].append(self.count[next_item])
        
    def add_multiple(self, result, next_items, for_item=0, session=0, position=None):   
        self.add(result, next_items[0], for_item, session=session, position=position)
    
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        
        recommendations = pd.DataFrame( self.recommendations )
        return ("Saver_next@" + str( self.length ) + ": "), 1, recommendations
        