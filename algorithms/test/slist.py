import numpy as np
import pandas as pd
import collections as col
import scipy
import os
import pickle

from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, csc_matrix, vstack
from time import time
from sklearn.preprocessing import normalize
from tqdm import tqdm


class SLIST:
    '''
    SLIST(reg=100)

    Parameters
    --------
    reg : int
        TODO(설명 추가). (Default value: 100)

    '''

    # 필수
    def __init__(self, reg=10, alpha=0.5, session_weight=-1, train_weight=-1, predict_weight=-1,
                 direction='part', normalize='l1', target_normalize='no', epsilon=10.0, train_size = 1.0,
                 remove_item='no', predict_by='order', model_path=None, session_key='SessionId', item_key='ItemId'):
        self.reg = reg
        self.normalize = normalize
        self.epsilon = epsilon
        self.train_size = train_size
        self.remove_item = remove_item
        self.predict_by = predict_by
        self.target_normalize = target_normalize
        self.alpha = alpha
        self.direction = direction  # 'all' or 'part'
        self.train_weight = float(train_weight)
        self.predict_weight = float(predict_weight)
        self.session_weight = session_weight*24*3600

        self.model_path = os.path.join('./model_ckpt', model_path) if model_path else model_path

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

        if self.model_path is not None and os.path.exists(self.model_path):
            print("model loading...")
            with open(self.model_path, 'rb') as f:
                self.enc_w = pickle.load(f)
            return
        
        if self.train_size < 1.0:
            # Cut
            sessions_train = int(self.n_sessions * self.train_size)
            keep = data.sort_values('Time', ascending=False).SessionId.unique()[:(sessions_train-1)]
            data = data[ np.in1d( data.SessionId, keep ) ]
            # stats
            sessionids = data[self.session_key].unique()
            self.n_sessions = len(sessionids)

        # ||X - XB||
        input1, target1, row_weight1 = self.make_train_matrix(data, weight_by='SLIS')
        # ||Y - ZB||
        input2, target2, row_weight2 = self.make_train_matrix(data, weight_by='SLIT')
        # alpha * ||X - XB|| + (1-alpha) * ||Y - ZB||
        input1.data = np.sqrt(self.alpha) * input1.data
        target1.data = np.sqrt(self.alpha) * target1.data
        input2.data = np.sqrt(1-self.alpha) * input2.data
        target2.data = np.sqrt(1-self.alpha) * target2.data

        input_matrix = vstack([input1, input2])
        target_matrix = vstack([target1, target2])
        w2 = row_weight1 + row_weight2  # list

        # P = (X^T * X + λI)^−1 = (G + λI)^−1
        # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
        # P =  G
        W2 = sparse.diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        print(f"G is made. Sparsity:{(1 - np.count_nonzero(G)/(self.n_items**2))*100}%")

        P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
        print("P is made")
        del G

        if self.epsilon < 10.0:
            C = -P @ (input_matrix.transpose().dot(W2).dot(input_matrix-target_matrix).toarray())

            mu = np.zeros(self.n_items)
            mu += self.reg
            mu_nonzero_idx = np.where(1 - np.diag(P)*self.reg + np.diag(C) >= self.epsilon)
            mu[mu_nonzero_idx] = (np.diag(1 - self.epsilon + C) / np.diag(P))[mu_nonzero_idx]

            # B = I - Pλ + C
            self.enc_w = np.identity(self.n_items, dtype=np.float32) - P @ np.diag(mu) + C
            print("weight matrix is made")
        else:
            self.enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()


        if self.model_path is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.enc_w, f, protocol=4)
            print("model saved")

    def make_train_matrix(self, data, weight_by='SLIT'):
        input_row = []
        target_row = []
        input_col = []
        target_col = []
        input_data = []
        target_data = []

        maxtime = data.Time.max()
        w2 = []
        sessionlengthmap = data['SessionIdx'].value_counts(sort=False)
        rowid = -1

        if weight_by == 'SLIT':
            if os.path.exists(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p'):
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p','rb') as f:
                    input_row, input_col, input_data, target_row, target_col, target_data, w2 = pickle.load(f)
            else:
                for sid, session in tqdm(data.groupby(['SessionIdx']), desc=weight_by):
                    slen = sessionlengthmap[sid]
                    # sessionitems = session['ItemIdx'].tolist() # sorted by itemid
                    sessionitems = session.sort_values(['Time'])['ItemIdx'].tolist()  # sorted by time
                    if self.remove_item == 'succession':
                        sessionitems = [s for i, s in enumerate(sessionitems) if i != (slen-1) and s != sessionitems[i+1]]
                    elif self.remove_item == 'repeat':
                        # 늦게 등장한 항목을 제거
                        sessionitems = sessionitems[::-1]
                        sessionitems = [s for i, s in enumerate(sessionitems) if i != (slen-1) and s not in sessionitems[i+1:]]
                        sessionitems = sessionitems[::-1]
                    else:
                        pass
                    slen = len(sessionitems)
                    if slen <= 1:
                        continue
                    stime = session['Time'].max()
                    w2 += [stime-maxtime] * (slen-1)
                    for t in range(slen-1):
                        rowid += 1
                        # input matrix
                        if self.direction == 'part':
                            input_row += [rowid] * (t+1)
                            input_col += sessionitems[:t+1]
                            for s in range(t+1):
                                input_data.append(-abs(t-s))
                            target_row += [rowid] * (slen - (t+1))
                            target_col += sessionitems[t+1:]
                            for s in range(t+1, slen):
                                target_data.append(-abs((t+1)-s))
                        elif self.direction == 'all':
                            input_row += [rowid] * slen
                            input_col += sessionitems
                            for s in range(slen):
                                input_data.append(-abs(t-s))
                            target_row += [rowid] * slen
                            target_col += sessionitems
                            for s in range(slen):
                                target_data.append(-abs((t+1)-s))
                        elif self.direction == 'sr':
                            input_row += [rowid]
                            input_col += [sessionitems[t]]
                            input_data.append(0)
                            target_row += [rowid] * (slen - (t+1))
                            target_col += sessionitems[t+1:]
                            for s in range(t+1, slen):
                                target_data.append(-abs((t+1)-s))
                        else:
                            raise ("You have to choose right 'direction'!")
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p','wb') as f:
                    pickle.dump([input_row, input_col, input_data, target_row, target_col, target_data, w2], f, protocol=4)
            input_data = list(np.exp(np.array(input_data) / self.train_weight))
            target_data = list(np.exp(np.array(target_data) / self.train_weight))
        elif weight_by == 'SLIS':
            if os.path.exists(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p'):
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p','rb') as f:
                    input_row, input_col, input_data, target_row, target_col, target_data, w2 = pickle.load(f)
            else:
                for sid, session in tqdm(data.groupby(['SessionIdx']), desc=weight_by):
                    rowid += 1
                    slen = sessionlengthmap[sid]
                    sessionitems = session['ItemIdx'].tolist()
                    stime = session['Time'].max()
                    # w2.append(np.exp((stime-maxtime)/self.session_weight))
                    w2.append(stime-maxtime)
                    input_row += [rowid] * slen
                    input_col += sessionitems

                target_row = input_row
                target_col = input_col
                input_data = np.ones_like(input_row)
                target_data = np.ones_like(target_row)
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p','wb') as f:
                    pickle.dump([input_row, input_col, input_data, target_row, target_col, target_data, w2], f, protocol=4)
        else:
            raise ("You have to choose right 'weight_by'!")

        # Use train_weight or not
        input_data = input_data if self.train_weight > 0 else list(np.ones_like(input_data))
        target_data = target_data if self.train_weight > 0 else list(np.ones_like(target_data))

        # Use session_weight or not
        w2 = list(np.exp(np.array(w2) / self.session_weight))
        w2 = w2 if self.session_weight > 0 else list(np.ones_like(w2))

        # Make sparse_matrix
        input_matrix = csr_matrix((input_data, (input_row, input_col)), shape=(max(input_row)+1, self.n_items), dtype=np.float32)
        target_matrix = csr_matrix((target_data, (target_row, target_col)), shape=input_matrix.shape, dtype=np.float32)
        print(f"[{weight_by}]sparse matrix {input_matrix.shape} is made.  Sparsity:{(1 - input_matrix.count_nonzero()/(self.n_items*input_matrix.shape[0]))*100}%")


        if weight_by == 'SLIT':
            pass
        elif weight_by == 'SLIS':
            # Value of repeated items --> 1
            input_matrix.data = np.ones_like(input_matrix.data)
            target_matrix.data = np.ones_like(target_matrix.data)

        # Normalization
        if self.normalize == 'l1':
            input_matrix = normalize(input_matrix, 'l1')
        elif self.normalize == 'l2':
            input_matrix = normalize(input_matrix, 'l2')
        else:
            pass

        if self.target_normalize == 'l1':
            target_matrix = normalize(target_matrix, 'l1')
        elif self.target_normalize == 'l2':
            target_matrix = normalize(target_matrix, 'l2')
        elif self.target_normalize == 'pop':
            target_matrix = normalize(target_matrix.T, 'l1').T
        else:
            pass

        return input_matrix, target_matrix, w2

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
            self.session_times = []

        if type == 'view':
            self.session_items.append(input_item_id)
            self.session_times.append(timestamp)

        # item id transfomration
        session_items_new_id = self.itemidmap[self.session_items].values
        predict_for_item_ids_new_id = self.itemidmap[predict_for_item_ids].values

        if skip:
            return

        W_test = np.ones_like(self.session_items, dtype=np.float32)
        W_test = self.enc_w[session_items_new_id[-1], session_items_new_id]
        # W_test = normalize(W_test.reshape(1,-1))
        if self.predict_by == 'identity':
            W_test = np.maximum(0, W_test)
        elif self.predict_by == 'tanh':
            W_test = np.maximum(0, np.tanh(W_test))
        elif self.predict_by == 'log':
            W_test = np.maximum(0, np.log(W_test)+1)
        elif self.predict_by == 'exp':
            W_test = np.maximum(0, np.exp(W_test/self.predict_weight)-1)
        elif self.predict_by == 'norm_exp':
            W_test = normalize(W_test.reshape(1, -1))
            W_test = np.maximum(0, np.exp(W_test/self.predict_weight)-1)
        elif self.predict_by == 'square':
            W_test = np.square(np.maximum(0, W_test))
        else:
            for i in range(len(W_test)):
                W_test[i] = np.exp(-abs(i+1-len(W_test))/self.predict_weight)

        W_test = W_test if self.predict_weight > 0 else np.ones_like(W_test)
        W_test = W_test.reshape(-1, 1)
        # print(W_test)

        # [session_items, num_items]
        preds = self.enc_w[session_items_new_id] * W_test
        # [num_items]
        preds = np.sum(preds, axis=0)
        preds = preds[predict_for_item_ids_new_id]

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()

        return series

    # 필수
    def clear(self):
        self.enc_w = {}
