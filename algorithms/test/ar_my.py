import numpy as np
from numpy.linalg.linalg import norm
import pandas as pd
import collections as col

from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize


class AR_MY:
    '''
    AR_MY()

    Parameters
    --------

    '''

    # 필수
    def __init__(self, self_count = True, session_key='SessionId', item_key='ItemId'):
        self.self_count = self_count
        self.session_key = session_key
        self.item_key = item_key

        # updated while recommending
        self.session = -1
        self.session_items = []

    # 필수
    def fit(self, data, test=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        # make new session ids(1 ~ #sessions)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.sessionidmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': self.sessionidmap[sessionids].values}), on=self.session_key, how='inner')

        # make new item ids(1 ~ #items)
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        #    SessionId          Time  ItemId        Date     Datestamp                             TimeO  ItemSupport  SessionIdx  ItemIdx
        # 0          1  1.462752e+09    9654  2016-05-09  1.462752e+09  2016-05-09 00:01:15.848000+00:00          399           0        0

        # sort by time
        data.sort_values(by='Time', axis=0, inplace=True)

        # sort by sessionIdx
        data.sort_values(by='SessionIdx', axis=0, inplace=True)

        # reset index (0 ~ #event)
        data.reset_index(drop=True, inplace=True)

        # item X item matrix
        self.ar_matrix = np.zeros((self.n_items, self.n_items))

        for sid, session in test.groupby([self.session_key]):
            sessionitems = session[self.item_key].tolist()
            for i in sessionitems:
                new_i = self.itemidmap[i]
                for j in sessionitems:
                    new_j = self.itemidmap[j]
                    self.ar_matrix[new_i, new_j] += 1
        
        if not self.self_count:
            for i in range(self.n_items):
                self.ar_matrix[i, i] = 0

        print("ar matrix is made")

    # 필수

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        # new session
        if session_id != self.session:
            self.session_items = []
            self.session = session_id

        if type == 'view':
            self.session_items.append(input_item_id)

        # item id transfomration
        session_items_new_id = self.itemidmap[self.session_items].values
        predict_for_item_ids_new_id = self.itemidmap[predict_for_item_ids].values

        if skip:
            return

        # [num_items]
        preds = self.ar_matrix[session_items_new_id[-1]]
        # [num_items]
        preds = preds[predict_for_item_ids_new_id]

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()

        return series

    # 필수
    def clear(self):
        self.enc_w = {}
