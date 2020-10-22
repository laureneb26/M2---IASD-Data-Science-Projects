from sklearn.base import BaseEstimator, RegressorMixin
from numba import jit, prange
import numpy as np
import scipy.sparse

class BasicMF(BaseEstimator, RegressorMixin):
    def __init__(self,k=10, l=0.1, m=0.1, solver='als', max_iter=100, tol=0.1, learning_rate=0.1, seed=None):
        self.k = k
        self.l = l
        self.m = m

        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

        if seed:
            np.random.seed(seed)
    
    def fit(self, X, y, shape=None, val=None):
        if shape:
            R = scipy.sparse.coo_matrix((y, (X[:,0], X[:,1])), shape=shape)
        else:
            R = scipy.sparse.coo_matrix((y, (X[:,0], X[:,1])))

        #self.I = np.random.normal(0, 2,(R.shape[0], self.k))
        #self.U = np.random.normal(0, 2,(R.shape[1], self.k))

        self.I = np.ones((R.shape[0], self.k))
        self.U =np.ones((R.shape[1], self.k))

        if self.solver=='als':
            next_UI = lambda U,I: BasicMF._ALS(R.tocsc(), R.tocsr(), U, I,self.l, self.m)
        elif self.solver=='gd':
            next_UI = lambda U,I:  BasicMF._GD(R.row, R.col, R.data, U, I, self.l, self.m, self.learning_rate)

        loss_f = lambda R,U,I : sum([ (r - np.dot(I[i,:], U[j,:]))**2 for i,j,r in zip(R.row, R.col, R.data)]) + self.l*np.linalg.norm(I) + self.m*np.linalg.norm(U)
        if val:
            X_test, R_test = val
            rmse_val = lambda : np.sqrt(((R_test - self.predict(X_test))**2).sum()/R_test.size)
            rmse_train = lambda : np.sqrt(((y - self.predict(X))**2).sum()/y.size)
        self.history = []
        for i in range(self.max_iter):
            loss = loss_f(R,self.U, self.I)
            if val:
                self.history.append([rmse_train(), rmse_val()])
            else:
                self.history.append(loss)
            if loss < self.tol:
                break
            self.U, self.I = next_UI(self.U,self.I)
        return self
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _GD(X_row, X_col, X_data, U, I, l, m, learning_rate):
        for k in prange(X_row.size):
            r, i, j = X_data[k], X_row[k], X_col[k]
            e_ij = r - np.dot(I[i,:],U[j,:])

            I_i = I[i, :][:]
            
            I[i, :] += learning_rate * (e_ij * U[j, :] - l * I[i,:])
            U[j, :] += learning_rate * (e_ij * I_i - m * U[j,:])
        return U, I
    
    # implémentation naïve décrite dans : Fast als-based matrix factorization for explicit and implicit feedback datasets.
    # https://www.academia.edu/download/37108248/2010_-_I._Pilaszy__D._Zibriczky__D._Tikk_-_ALS1.pdf
    # D'autre implémentation plus efficace existe notamment : Fast Matrix Factorization for Online Recommendation with Implicit Feedback
    # https://arxiv.org/pdf/1708.05024.pdf
    @staticmethod
    def _ALS(Ri, Rj, U, I, l, m):
        k = I.shape[1]
        for i in range(U.shape[0]):
            ru = Ri[:,i]
            nu = (ru.nonzero()[0])
            Iu = I[nu,:]
            Au = Iu.T.dot(Iu)
            du = Iu.T@ru[nu,:]
            U[i] = np.linalg.solve(Au + m*np.eye(k),du)[:,0]
        for j in range(I.shape[0]):
            ri = Rj[j,:].T
            ni = (ri.nonzero()[0])
            Ui = U[ni,:]
            Ai = Ui.T.dot(Ui)
            di = Ui.T@ri[ni,:]
            I[j] = np.linalg.solve(Ai + l*np.eye(k),di)[:,0]
        return U,I


    def predict(self, X, y=None):
        #R = self.I@self.U.T
        return np.array([self.I[i].dot(self.U[j]) for i,j in X])
    
    def get_params(self, deep=True):
        return {"k": self.k, "l": self.l, "m": self.m, "solver": self.solver, "max_iter": self.max_iter, "tol": self.tol, "learning_rate": self.learning_rate}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


