import numpy as np

class Count_leq: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        Count@length
    '''
    
    def __init__(self, length=20, threshold=5):
        self.length = length;
        self.threshold = threshold;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        
        self.test_popbin = {}
        self.test_position = {}
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        self.test += 1

        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
            self.test_position[position] += 1         
        
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
        return ("Count@" + str(self.length) + "<=" + str(self.threshold) + ": "), (self.test), self.result_pop_bin(), self.result_position()

    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nCount@' + str(self.length) + "<=" + str(self.threshold) + ': ;'
        for key in self.test_popbin:
            csv += str( self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nCount@' + str(self.length) + "<=" + str(self.threshold) + ': ;'
        for key in self.test_position:
            csv += str( self.test_position[key] ) + ';'
            
        return csv


class MRR_leq: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average mean reciprocal rank for a result list with the defined length. 

    Parameters
    -----------
    length : int
        MRR@length
    '''
    def __init__(self, length=20, threshold=5):
        self.length = length;
        self.threshold = threshold
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.pos=0
        
        self.test_popbin = {}
        self.pos_popbin = {}
        
        self.test_position = {}
        self.pos_position = {}
    
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        # position start at 0
        if position >= self.threshold:
            return

        res = result[:self.length]
        

        
        self.test += 1
        
        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
                self.pos_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
                self.pos_position[position] = 0
            self.test_position[position] += 1
        
        if next_item in res.index:
            rank = res.index.get_loc( next_item )+1
            self.pos += ( 1.0/rank )
            
            if pop_bin is not None:
                self.pos_popbin[pop_bin] += ( 1.0/rank )
            
            if position is not None:
                self.pos_position[position] += ( 1.0/rank )
                   
        
        
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
        return ("MRR@" + str(self.length) + "<=" + str(self.threshold) + "<=" + str(self.threshold) + ": "), (self.pos/self.test), self.result_pop_bin(), self.result_position()
    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nPrecision@' + str(self.length) + "<=" + str(self.threshold) + ': ;'
        for key in self.test_popbin:
            csv += str( self.pos_popbin[key] / self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nPrecision@' + str(self.length) + "<=" + str(self.threshold) + ': ;'
        for key in self.test_position:
            csv += str( self.pos_position[key] / self.test_position[key] ) + ';'
            
        return csv
    
class HitRate_leq: 
    '''
    MRR( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20, threshold=5):
        self.length = length;
        self.threshold = threshold;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
        
        self.test_popbin = {}
        self.hit_popbin = {}
        
        self.test_position = {}
        self.hit_position = {}
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        self.test += 1

        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
                self.hit_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
                self.hit_position[position] = 0
            self.test_position[position] += 1
                
        if next_item in result[:self.length].index:
            self.hit += 1
            
            if pop_bin is not None:
                self.hit_popbin[pop_bin] += 1
            
            if position is not None:
                self.hit_position[position] += 1
            
        
        
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
        return ("HitRate@" + str(self.length) + "<=" + str(self.threshold) + ": "), (self.hit/self.test), self.result_pop_bin(), self.result_position()

    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nHitRate@' + str(self.length) + "<=" + str(self.threshold) + ': ;'
        for key in self.test_popbin:
            csv += str( self.hit_popbin[key] / self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nHitRate@' + str(self.length) + "<=" + str(self.threshold) + ': ;'
        for key in self.test_position:
            csv += str( self.hit_position[key] / self.test_position[key] ) + ';'
            
        return csv

class Precision_leq: 
    '''
    Precision( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20, threshold=5):
        self.length = length;
        self.threshold = threshold;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
    
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        self.test += self.length
        self.hit += len( set(next_item) & set(result[:self.length].index) )
    
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        self.test += 1
        self.hit += len( set(next_items) & set(result[:self.length].index) ) / self.length
        
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
        return ("Precision@" + str(self.length) + "<=" + str(self.threshold) + ": "), (self.hit/self.test)
    
class Recall_leq: 
    '''
    Precision( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20, threshold=5):
        self.length = length;
        self.threshold = threshold;

    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
    
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        self.test += 1
        a=set(next_item)
        b=set(result[:self.length].index)
        c=set(next_item) & set(result[:self.length].index)
        self.hit += len( set(next_item) & set(result[:self.length].index) )
       
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        self.test += 1
        self.hit += len( set(next_items) & set(result[:self.length].index) ) / len(next_items)
        
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
        return ("Recall@" + str(self.length) + "<=" + str(self.threshold) + ": "), (self.hit/self.test)

class MAP_leq: 
    '''
    MAP( length=20 )

    Used to iteratively calculate the mean average precision for a result list with the defined length. 

    Parameters
    -----------
    length : int
        MAP@length
    '''
    def __init__(self, length=20, threshold=5):
        self.length = length;
        self.threshold = threshold;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.pos=0
    
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        
        last_recall = 0
        
        res = 0
        
        for i in range(self.length):
            recall = self.recall(result[:i].index, next_items)
            precision = self.precision(result[:i].index, next_items)
            res += precision * (recall - last_recall)
            last_recall = recall
        
        self.pos += res
        self.test += 1
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        if position >= self.threshold:
            return
        sum = 0
        
        for i in range(self.length):
            sum += self.mrr(result, next_item, i+1)
        
        self.pos += ( sum / self.length )
        self.test += 1
    
    def recall( self, result, next_items ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        return len( set(next_items) & set(result) ) / len( next_items )
    
    def precision( self, result, next_items ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        return len( set(next_items) & set(result) ) / self.length
    
    def mrr( self, result, next_item, n ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        res = result[:n]
        
        if next_item in res.index:
            rank = res.index.get_loc( next_item )+1
            return 1.0/rank
        else:
            return 0
        
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
        return ("MAP@" + str(self.length) + "<=" + str(self.threshold) + ": "), (self.pos/self.test)