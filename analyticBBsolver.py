import numpy as np
from numpy.linalg import inv
import numpy.linalg as linalg



class LLSQsolver:
    
    def __init__(self,degrees,poles,cov,km,combined):
        
        self.degrees = np.asarray(degrees)
        self.poles = np.asarray(poles)
        self.cov = np.asarray(cov)
        self.kmfull = np.asarray(km)
        self.numbb=int(self.degrees.size)
        self.numpoles=int(self.poles.size)
        self.covinv = inv(cov)
        self.fac = 1
        if combined:
            self.fac = 2
            #self.degrees = np.concatenate([self.degrees,self.degrees])
        self.ksize = int(self.kmfull.size/self.numpoles/self.fac)
        self.makeH()
        

        
    def makeH(self):
        #self.kmfull = [self.km]*self.numpoles
        self.H = np.zeros((self.numpoles*self.ksize*self.fac,self.numpoles*self.numbb*self.fac))
        for i in range(0,self.numpoles*self.fac):
            for j in range(0,self.numbb):
                self.H[i*self.ksize:(i+1)*self.ksize,j+(i*self.numbb)] = self.kmfull[i]**self.degrees[j]

        
        self.Ht = self.H.transpose()
        
        
        
    def polysolve(self,res):
        C1 = linalg.pinv((np.matmul(self.Ht,np.matmul(self.covinv,self.H))))
        C2 = np.matmul(self.Ht,np.matmul(self.covinv,res))
        theta = np.matmul(C1,C2)
        return(theta)
        
        
    def BBk(self,res):
        Al = self.polysolve(res)
        BB = np.zeros((self.numpoles*self.fac,self.ksize))
        for i in range(0,self.numpoles*self.fac):
            for a,n in enumerate(self.degrees):
                BB[i,:] += Al[a+i*self.numbb]*self.kmfull[i]**n
             
        return BB
                    

    
