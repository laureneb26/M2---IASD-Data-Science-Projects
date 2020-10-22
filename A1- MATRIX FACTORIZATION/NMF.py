from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import scipy.sparse

class NMF(BaseEstimator, RegressorMixin):
    def __init__(self,k=10, solver='wnmf', max_iter=100, n_steps=1, cooldown=0, minValue=1, seed=None):
        self.k = k
        self.solver = solver
        self.max_iter = max_iter

        self.n_steps = n_steps
        self.cooldown = 0
        self.minValue = minValue

        self.seed = seed
        
    def fit(self, X, y, shape=None, val=None, ech=10):
        if shape:
            R = scipy.sparse.coo_matrix((y, (X[:,0], X[:,1])), shape=shape).todense()
        else:
            R = scipy.sparse.coo_matrix((y, (X[:,0], X[:,1]))).todense()

        self.U = self._randomMatrixInit((R.shape[0], self.k))
        self.V = self._randomMatrixInit((R.shape[1], self.k))

        if self.solver=='wnmf':
            W = np.zeros_like(R)
            W[R.nonzero()] = 1
            next_UV = lambda U,V,_: self._wmnf1t(U,V,W,R)
        elif self.solver=='em':
            R = self._completeMatrix(R)
            next_UV = lambda U,V,R : self._EM1t(U,V,R,n_steps=self.n_steps)
        
        self.history = []
        inc = 0
        for i in range(self.max_iter):
            self.U, self.V = next_UV(self.U, self.V.T, R)
            inc+=1
            if inc==ech:
                if val:
                    self.history.append([self._MAE(X,y,self.U.dot(self.V.T)), self._MAE(val[0], val[1],self.U.dot(self.V.T))])
                else:
                    self.history.append(self._MAE(X,y,self.U.dot(self.V.T)))
                inc=0
            
            if(self.solver == 'em'):
                R = np.copy(self.U@self.V.T)
                if self.cooldown!=0:
                    m_steps = max(self.minValue, m_steps * self.cooldown)
                for index, point in enumerate(X):
                    R[point[0],point[1]] = y[index]
        return self
    
    #Application d'une seul itération de la WNMF
    def _wmnf1t(self, P, Q, W, R):
        A = np.multiply(W,R).dot(Q.T)
        B = (np.multiply(W,P@Q)).dot(Q.T)
        C =(P.T).dot(np.multiply(W,R))
        D = (P.T).dot(np.multiply(W,(P@Q)))
        
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if B[i,j]==0:
                    P[i,j]=0
                else:
                    P[i,j] *= A[i,j]/B[i,j]
                
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                if D[i,j]==0:
                    Q[i,j] = 0
                else:
                    Q[i,j] *= C[i,j]/D[i,j]
        return P, Q.T
    
    #Application d'un lot d'itération de l'algorithme EM
    def _EM1t(self, U,V,R, n_steps=100):
        #step M
        A = R.dot(V.T)
        B = U.dot(V.dot(V.T))
        C = U.T.dot(R)
        D = U.T.dot(U.dot(V))
        
        for k in range(n_steps):
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    if(B[i,j]==0):
                        U[i,j]=0
                    else:
                        U[i,j] *= (A[i,j]/B[i,j])
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    if(D[i,j]==0):
                        V[i,j]=0
                    else:
                        V[i,j] *= (C[i,j]/D[i,j])
        return U, V.T
    
    def _completeMatrix(self, R):
        X = np.copy(R)
        for i in range(X.shape[0]):
            k=0
            v=0
            for j in range(X.shape[1]):
                if X[i,j]!=0:
                    k+=1
                    v+=X[i,j]
            if k!=0:
                v = round(v/k,1)
            for j in range(X.shape[1]):
                if X[i,j]==0:
                    X[i,j]= v
        return X

    def predict(self, X, y=None):
        #R = self.I@self.U.T
        return np.array([self.I[i].dot(self.U[j]) for i,j in X])

    def _MAE(self, X,y,R2):
        error = 0
        k = 0
        for i,point in enumerate(X):
            error += abs(y[i] - R2[point[0],point[1]])
            k+=1
        error = error/k
        return error
    
    def _RMSE(self, X,y,R2):
        error = 0
        k = 0
        for i,point in enumerate(X):
            error += (y[i] - R2[point[0],point[1]])**2
            k+=1
        error = math.sqrt(error/k)
        return error

    def _randomMatrixInit(self, shape):
        if self.seed:
            np.random.seed(self.seed)
        return 2*(np.ones(shape) - np.random.rand(shape[0],shape[1]))