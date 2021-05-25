import numpy as np
import shared
import scipy.integrate as integrate



class modelnoWl02:

        
        def __init__(self, params,combined):

                
                self.B = params[0]
                self.alpha_perp = params[1]
                self.alpha_par = params[2]
                self.beta = shared.f/self.B
                #self.fn = params[3]
                #self.beta = self.fn/self.B
                #self.sigpar = params[4]
                #self.sigperp = params[5]
                #self.sigs = params[6]
                
                self.sigpar = shared.sigpar
                self.sigperp = shared.sigperp
                self.sigs = shared.sigs
                
                if combined:
                        self.B2 =params[3]
                        self.beta2 = f/self.B2
                        #self.B2 = params[4]
                        #self.fn2 = params[5]
                        #self.beta2 = self.B2/self.fn

                self.F = self.alpha_par / self.alpha_perp

                self.combined = combined
                
                self.muobs = shared.muobs
                self.Olin = shared.Olin
                self.Psmfit = shared.Psmfit
                
                self.L0 = shared.L0
                self.L2 = shared.L2
                self.L4 = shared.L4


        def kprime(self,k):

                kp = k/self.alpha_perp * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**0.5
                return kp

        def muprime(self):

                mup = self.muobs/self.F * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**(-0.5)
                return mup

        def Pkmuf(self,kobs):

                combined = self.combined

                mup = self.muprime()
                Pkmuint = []

                if combined:
                        cap =1
                        for k in kobs[0:modelhalf]:
                                kp = self.kprime(k)
                                Psmkmu1 = self.Psmkmuf(mup,kp,k,cap)

                                Pkmu1 = Psmkmu1 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu1)
                        cap=2
                        for k in kobs[modelhalf:modelsize]:
                                kp = self.kprime(k)
                                Psmkmu2 = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu2 = Psmkmu2 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))

                                Pkmuint.append(Pkmu2)	

                        return np.asarray(Pkmuint)
        
                else:
                        cap =1
                        for k in kobs:
                                kp = self.kprime(k)
                                Psmkmu = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu = Psmkmu * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 *self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu)
                        return np.asarray(Pkmuint)

        def Psmkmuf(self,mup,kp,k,cap):
                R = 1.0
                if cap ==1:
                        Pskmu = (self.B**2) * (1+self.beta*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                if cap ==2:
                        Pskmu = (self.B2**2) * (1+self.beta2*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                return Pskmu


        def Ffogf(self,mu,k):
        #Ffog = 1.0/(1+(k**2 * mu**2 * sigs**2)/2)**2
                Ffog = 1.0/(1+((k*mu*self.sigs)**2)/2)
                return Ffog


        def run(self,k):

                combined = self.combined
                if combined:
                        Pkmu = self.Pkmuf(k)
                        Pkmu1 = Pkmu[0:modelhalf]
                        Pkmu2 = Pkmu[modelhalf:modelsize]

                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        integrand20 = Pkmu2*self.L0
                        integrand22 = Pkmu2*self.L2
                        integrand24 = Pkmu2*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand12,x=muobs,axis=1)
                        #P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand14,x=muobs,axis=1)

                        P_2_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand20,x=muobs,axis=1)
                        P_2_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand22,x=muobs,axis=1)
                        #P_2_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand24,x=muobs,axis=1)	


                        Pkml = np.concatenate([P_1_0,P_1_2,P_2_0,P_2_2])
                      
                        res = Pkml-WPkm_cut
                        
                        BB = solver.BBk(res).flatten()
                       
                        Pkmodel = Pkml + BB
                        
                        return Pkmodel


                else:

                        Pkmu1 = self.Pkmuf(k)
                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=self.muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand12,x=self.muobs,axis=1)
                        #P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand14,x=self.muobs,axis=1)
    
                        Pkml = np.concatenate([P_1_0,P_1_2])

                        res = shared.Pkdata-Pkml
                        BB = shared.solver.BBk(res).flatten()
                        
                        Pkmodel = Pkml + BB

                        return Pkmodel
                    
                    
                    
                    
                    
                    
class modelnoWl024:

        
        def __init__(self, params,combined):

                
                self.B = params[0]
                self.alpha_perp = params[1]
                self.alpha_par = params[2]
                self.beta = shared.f/self.B
                #self.fn = params[3]
                #self.beta = params[3]
                #self.sigpar = params[4]
                #self.sigperp = params[5]
                #self.sigs = params[6]
                
                self.sigpar = shared.sigpar
                self.sigperp = shared.sigperp
                self.sigs = shared.sigs
                
                self.sigsmooth = 15


                if combined:
                        self.B2 =params[3]
                        self.beta2 = shared.f/self.B2
                        #self.B2 = params[4]
                        #self.fn2 = params[5]
                        #self.beta2 = self.B2/self.fn

                self.F = self.alpha_par / self.alpha_perp


                
                self.combined = combined
                self.muobs = shared.muobs
                self.Olin = shared.Olin
                self.Psmfit = shared.Psmfit
                
                self.modelhalf = shared.modelhalf
                self.modelsize = shared.modelsize
                
                self.L0 = shared.L0
                self.L2 = shared.L2
                self.L4 = shared.L4


        def kprime(self,k):

                kp = k/self.alpha_perp * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**0.5
                return kp

        def muprime(self):

                mup = self.muobs/self.F * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**(-0.5)
                return mup

        def Pkmuf(self,kobs):

                combined = self.combined

                mup = self.muprime()
                Pkmuint = []

                if combined:
                        cap =1
                        for k in kobs[0:self.modelhalf]:
                                kp = self.kprime(k)
                                Psmkmu1 = self.Psmkmuf(mup,kp,k,cap)

                                Pkmu1 = Psmkmu1 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu1)
                        cap=2
                        for k in kobs[self.modelhalf:self.modelsize]:
                                kp = self.kprime(k)
                                Psmkmu2 = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu2 = Psmkmu2 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))

                                Pkmuint.append(Pkmu2)	

                        return np.asarray(Pkmuint)
        
                else:
                        cap =1
                        for k in kobs:
                                kp = self.kprime(k)
                                Psmkmu = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu = Psmkmu * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 *self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu)
                        return np.asarray(Pkmuint)

        def Psmkmuf(self,mup,kp,k,cap):
                #R = 1.0-np.exp(-0.5*(kp*self.sigsmooth)**2)
                R = 1.0
                if cap ==1:
                        Pskmu = (self.B**2) * (1+self.beta*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                if cap ==2:
                        Pskmu = (self.B2**2) * (1+self.beta2*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                return Pskmu


        def Ffogf(self,mu,k):
        #Ffog = 1.0/(1+(k**2 * mu**2 * sigs**2)/2)**2
                Ffog = 1.0/(1+((k*mu*self.sigs)**2)/2)
                return Ffog


        def run(self,k):

                combined = self.combined
                if combined:
                        Pkmu = self.Pkmuf(k)
                        Pkmu1 = Pkmu[0:self.modelhalf]
                        Pkmu2 = Pkmu[self.modelhalf:self.modelsize]

                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        integrand20 = Pkmu2*self.L0
                        integrand22 = Pkmu2*self.L2
                        integrand24 = Pkmu2*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=self.muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand12,x=self.muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand14,x=self.muobs,axis=1)

                        P_2_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand20,x=self.muobs,axis=1)
                        P_2_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand22,x=self.muobs,axis=1)
                        P_2_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand24,x=self.muobs,axis=1)


                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4,P_2_0,P_2_2,P_2_4])
                      
                        res = shared.Pkdata-Pkml
                        BB = shared.solver.BBk(res).flatten()
                        
                        Pkmodel = Pkml + BB
                        
                        return Pkmodel


                else:

                        Pkmu1 = self.Pkmuf(k)
                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=self.muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand12,x=self.muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand14,x=self.muobs,axis=1)
    
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4])

                        res = shared.Pkdata-Pkml
                        BB = shared.solver.BBk(res).flatten()
                        
                        Pkmodel = Pkml + BB

                        return Pkmodel
        
        


class modelWl02:

        
        def __init__(self, params,combined):

                
                self.B = params[0]
                self.alpha_perp = params[1]
                self.alpha_par = params[2]
                self.beta = shared.f/self.B
                #self.fn = params[3]
                #self.beta = self.fn/self.B
                #self.sigpar = params[4]
                #self.sigperp = params[5]
                #self.sigs = params[6]
                
                self.sigpar = shared.sigpar
                self.sigperp = shared.sigperp
                self.sigs = shared.sigs
                
                if combined:
                        self.B2 =params[3]
                        self.beta2 = f/self.B2
                        #self.B2 = params[4]
                        #self.fn2 = params[5]
                        #self.beta2 = self.B2/self.fn

                self.F = self.alpha_par / self.alpha_perp

                self.combined = combined
                
                self.muobs = shared.muobs
                self.Olin = shared.Olin
                self.Psmfit = shared.Psmfit
                
                self.L0 = shared.L0
                self.L2 = shared.L2
                self.L4 = shared.L4


        def kprime(self,k):

                kp = k/self.alpha_perp * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**0.5
                return kp

        def muprime(self):

                mup = self.muobs/self.F * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**(-0.5)
                return mup

        def Pkmuf(self,kobs):

                combined = self.combined

                mup = self.muprime()
                Pkmuint = []

                if combined:
                        cap =1
                        for k in kobs[0:modelhalf]:
                                kp = self.kprime(k)
                                Psmkmu1 = self.Psmkmuf(mup,kp,k,cap)

                                Pkmu1 = Psmkmu1 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu1)
                        cap=2
                        for k in kobs[modelhalf:modelsize]:
                                kp = self.kprime(k)
                                Psmkmu2 = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu2 = Psmkmu2 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))

                                Pkmuint.append(Pkmu2)	

                        return np.asarray(Pkmuint)
        
                else:
                        cap =1
                        for k in kobs:
                                kp = self.kprime(k)
                                Psmkmu = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu = Psmkmu * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 *self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu)
                        return np.asarray(Pkmuint)

        def Psmkmuf(self,mup,kp,k,cap):
                R = 1.0
                if cap ==1:
                        Pskmu = (self.B**2) * (1+self.beta*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                if cap ==2:
                        Pskmu = (self.B2**2) * (1+self.beta2*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                return Pskmu


        def Ffogf(self,mu,k):
        #Ffog = 1.0/(1+(k**2 * mu**2 * sigs**2)/2)**2
                Ffog = 1.0/(1+((k*mu*self.sigs)**2)/2)
                return Ffog


        def run(self,k):

                combined = self.combined
                if combined:
                        Pkmu = self.Pkmuf(k)
                        Pkmu1 = Pkmu[0:modelhalf]
                        Pkmu2 = Pkmu[modelhalf:modelsize]

                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        integrand20 = Pkmu2*self.L0
                        integrand22 = Pkmu2*self.L2
                        integrand24 = Pkmu2*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand12,x=muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand14,x=muobs,axis=1)

                        P_2_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand20,x=muobs,axis=1)
                        P_2_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand22,x=muobs,axis=1)
                        P_2_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand24,x=muobs,axis=1)	


                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4,P_2_0,P_2_2,P_2_4])

                        WPkm = np.dot(W,np.dot(M,Pkml))
                        newmod = np.reshape(WPkm,(10,40))
                        P_1_0 = newmod[0,2:23]
                        P_1_2 = newmod[2,2:23]
                        #P_1_4 = newmod[4,2:23]
                        P_2_0 = newmod[5,2:23]
                        P_2_2 = newmod[7,2:23]
                        #P_2_4 = newmod[9,2:23]
                           
                        Pkml = np.concatenate([P_1_0,P_1_2,P_2_0,P_2_2])
                            
                        res = Pkml-WPkm_cut
                        
                        BB = solver.BBk(res).flatten()
                       
                        Pkmodel = Pkml + BB
                        
                        return Pkmodel


                else:

                        Pkmu1 = self.Pkmuf(k)
                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=self.muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand12,x=self.muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand14,x=self.muobs,axis=1)
    
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4])

                        convolved_model = np.dot(shared.W,np.dot(shared.M,Pkml))
                        newmod = np.reshape(convolved_model,(5,40)) 
                        P_1_0 = newmod[0,int(shared.kmin/0.01):int(shared.kmax/shared.dk)]
                        P_1_2 = newmod[2,int(shared.kmin/0.01):int(shared.kmax/shared.dk)]
                        #P_1_4 = newmod[4,int(shared.kmin/0.01):int(shared.kmax/shared.dk)]
                        
        
                        Pkml = np.concatenate([P_1_0,P_1_2])

                        res = shared.Pkdata-Pkml
                        BB = shared.solver.BBk(res).flatten()
                        
                        Pkmodel = Pkml + BB

                        return Pkmodel
        
        
        
class modelWl024:

        
        def __init__(self, params,combined):

                
                self.B = params[0]
                self.alpha_perp = params[1]
                self.alpha_par = params[2]
                self.beta = shared.f/self.B
                #self.fn = params[3]
                #self.beta = self.fn/self.B
                #self.sigpar = params[4]
                #self.sigperp = params[5]
                #self.sigs = params[6]
                
                self.sigpar = shared.sigpar
                self.sigperp = shared.sigperp
                self.sigs = shared.sigs
                
                if combined:
                        self.B2 =params[3]
                        self.beta2 = f/self.B2
                        #self.B2 = params[4]
                        #self.fn2 = params[5]
                        #self.beta2 = self.B2/self.fn

                self.F = self.alpha_par / self.alpha_perp

                self.combined = combined
                
                self.muobs = shared.muobs
                self.Olin = shared.Olin
                self.Psmfit = shared.Psmfit
                
                self.L0 = shared.L0
                self.L2 = shared.L2
                self.L4 = shared.L4


        def kprime(self,k):

                kp = k/self.alpha_perp * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**0.5
                return kp

        def muprime(self):

                mup = self.muobs/self.F * (1.0 + self.muobs**2 * (1.0/self.F**2 - 1.0))**(-0.5)
                return mup

        def Pkmuf(self,kobs):

                combined = self.combined

                mup = self.muprime()
                Pkmuint = []

                if combined:
                        cap =1
                        for k in kobs[0:modelhalf]:
                                kp = self.kprime(k)
                                Psmkmu1 = self.Psmkmuf(mup,kp,k,cap)

                                Pkmu1 = Psmkmu1 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu1)
                        cap=2
                        for k in kobs[modelhalf:modelsize]:
                                kp = self.kprime(k)
                                Psmkmu2 = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu2 = Psmkmu2 * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 * self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))

                                Pkmuint.append(Pkmu2)	

                        return np.asarray(Pkmuint)
        
                else:
                        cap =1
                        for k in kobs:
                                kp = self.kprime(k)
                                Psmkmu = self.Psmkmuf(mup,kp,k,cap)
                                Pkmu = Psmkmu * (1+ (self.Olin(kp) -1) * np.exp(-1*(kp**2 * mup**2 *self.sigpar**2 + kp**2*(1-mup**2)*self.sigperp**2)/2.0))
                                Pkmuint.append(Pkmu)
                        return np.asarray(Pkmuint)

        def Psmkmuf(self,mup,kp,k,cap):
                R = 1.0
                if cap ==1:
                        Pskmu = (self.B**2) * (1+self.beta*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                if cap ==2:
                        Pskmu = (self.B2**2) * (1+self.beta2*mup**2 *R)**2 * self.Psmfit(k) * self.Ffogf(mup,kp)

                return Pskmu


        def Ffogf(self,mu,k):
        #Ffog = 1.0/(1+(k**2 * mu**2 * sigs**2)/2)**2
                Ffog = 1.0/(1+((k*mu*self.sigs)**2)/2)
                return Ffog


        def run(self,k):

                combined = self.combined
                if combined:
                        Pkmu = self.Pkmuf(k)
                        Pkmu1 = Pkmu[0:modelhalf]
                        Pkmu2 = Pkmu[modelhalf:modelsize]

                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        integrand20 = Pkmu2*self.L0
                        integrand22 = Pkmu2*self.L2
                        integrand24 = Pkmu2*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand12,x=muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand14,x=muobs,axis=1)

                        P_2_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand20,x=muobs,axis=1)
                        P_2_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand22,x=muobs,axis=1)
                        P_2_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand24,x=muobs,axis=1)	


                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4,P_2_0,P_2_2,P_2_4])

                        WPkm = np.dot(W,np.dot(M,Pkml))
                        newmod = np.reshape(WPkm,(10,40))
                        P_1_0 = newmod[0,2:23]
                        P_1_2 = newmod[2,2:23]
                        P_1_4 = newmod[4,2:23]
                        P_2_0 = newmod[5,2:23]
                        P_2_2 = newmod[7,2:23]
                        P_2_4 = newmod[9,2:23]
                           
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4,P_2_0,P_2_2,P_2_4])
                            
                        res = Pkml-WPkm_cut
                        
                        BB = solver.BBk(res).flatten()
                       
                        Pkmodel = Pkml + BB
                        
                        return Pkmodel


                else:

                        Pkmu1 = self.Pkmuf(k)
                        integrand10 = Pkmu1*self.L0
                        integrand12 = Pkmu1*self.L2
                        integrand14 = Pkmu1*self.L4

                        P_1_0 = 0.5 * 1./(self.alpha_perp**2 * self.alpha_par)* integrate.simps(integrand10,x=self.muobs,axis=1)
                        P_1_2 = 5./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand12,x=self.muobs,axis=1)
                        P_1_4 = 9./2 * 1./(self.alpha_perp**2 * self.alpha_par)*integrate.simps(integrand14,x=self.muobs,axis=1)
    
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4])

                        convolved_model = np.dot(shared.W,np.dot(shared.M,Pkml))
                        newmod = np.reshape(convolved_model,(5,40)) 
                        P_1_0 = newmod[0,int(shared.kmin/0.01):int(shared.kmax/shared.dk)]
                        P_1_2 = newmod[2,int(shared.kmin/0.01):int(shared.kmax/shared.dk)]
                        P_1_4 = newmod[4,int(shared.kmin/0.01):int(shared.kmax/shared.dk)]
                        
        
                        Pkml = np.concatenate([P_1_0,P_1_2,P_1_4])

                        res = shared.Pkdata-Pkml
                        BB = shared.solver.BBk(res).flatten()
                        
                        Pkmodel = Pkml + BB

                        return Pkmodel
        


