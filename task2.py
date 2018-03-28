from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm


        
class VolPlotter:
    def __init__(self, title, xlabel, ylabel):
        
        self.fig = plt.figure()
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_title(title) 
        
        self.ax.set_xlabel(xlabel)
        
        self.ax.set_ylabel(ylabel)
        
        
        
    def plot(self, xx, yy, matrix):
        
        x,y = np.meshgrid(xx,yy)
        
        CS = plt.contour(x,y,matrix,colors = ('indigo','purple','b','m','violet','aqua'), linewidths = 1.1)
        
        return self.ax.plot_surface(x, y, matrix, alpha = 0.6, cmap = cm.coolwarm, linewidth = 0.4)
    
    

    def show(self):
        
        pylab.show()
        
class poisson:
    
    def __init__(self,M,N,cell,density,iterations,ledge,redge,up,down,Epsilon,plotter):
        
        self.n = N
        
        self.m = M
        
        self.c = cell
        
        self.dp = density
        
        self.ir = iterations
        
        self.le = ledge
        
        self.re = redge
        
        self.up = up
        
        self.dn = down
        
        self.eps = Epsilon
        
        self.plotter = plotter
        
        
        
    def standart(self, amount):
        
        x = np.linspace(0.,self.n*self.c,self.n)
        
        y = np.linspace(0.,self.m*self.c,self.m)
        
        A = np.zeros((self.m, self.n))
        
        AFourier = np.zeros((self.m, self.n))
        
        for k in range(amount):
            
            for i in range(self.n):
                
                for j in range(self.m):
                    
                    AFourier[j,i] = 400./(np.pi*(2*k+1))*(np.cosh(np.pi/self.n*(2*k+1)*i) - 1/np.tanh(np.pi*(2*k+1))*np.sinh(np.pi/self.n*(2*k+1)*i))*np.sin(np.pi/self.m*(2*k+1)*j)
                    
            A = A + AFourier
            
        plotter.plot(x,y,A)
        
        return (A)
        
    def Jacoby(self):
        
        x = np.linspace(0.,self.n*self.c,self.n)
        
        y = np.linspace(0.,self.m*self.c,self.m)
        
        A=np.zeros((self.m, self.n))
        
        A[0,:] = self.up
        
        A[-1,:] = self.dn
        
        A[:,0] = self.le
        
        A[:,-1] = self.re
        
        for i in range(self.ir):
            
            B = 0.25*( A[0:-2,1:-1] + A[1:-1,0:-2] + A[1:-1,2:] + A[2:,1:-1]) + np.pi*self.dp[1:-1,1:-1]*self.c**2
            
            A[1:-1,1:-1] = B
        
        plotter.plot(x,y,A)
        
        return (A)
    
    
    
    def Gauss(self):
        
        x = np.linspace(0.,self.n*self.c,self.n)
        
        y = np.linspace(0.,self.m*self.c,self.m)
        
        A = np.zeros((self.m, self.n))
        
        A[0,:] = self.up
        
        A[-1,:] = self.dn
        
        A[:,0] = self.le
        
        A[:,-1] = self.re
        
        for k in range(self.ir):
            
            Sp1 = np.trace(A)
            
            for i in range(self.m-2):
                
                for j in range(self.n-2):
                    
                    A[i+1,j+1]=0.25*(A[i,j+1]+A[i+1,j]+A[i+1,j+2]+A[i+2,j+1]) + np.pi*self.dp[i+1,j+1]*self.c**2
                    
            Sp2 = np.trace(A)
            
            delta = np.absolute(Sp2 - Sp1)
            
            if (delta<=self.eps):
                
                break
                
        plotter.plot(x,y,A)
        
        return (A)
    
    
    
    def Relax(self,omega):
        
            x = np.linspace(0.,self.n*self.c,self.n)
            
            y = np.linspace(0.,self.m*self.c,self.m)
            
            A = np.zeros((self.m, self.n))
            
            R = np.zeros((self.m, self.n))
            
            A[0,:] = self.up
            
            A[-1,:] = self.dn
            
            A[:,0] = self.le
            
            A[:,-1] = self.re
            
            for i in range(self.ir):
                
                Sp1 = np.trace(A)
                
                B = 0.25*( A[0:-2,1:-1] + A[1:-1,0:-2] + A[1:-1,2:] + A[2:,1:-1]) + np.pi*self.dp[1:-1,1:-1]*self.c**2
                
                R[1:-1,1:-1] = B-A[1:-1,1:-1]
                
                A = A + omega*R
                
                Sp2 = np.trace(A)
                
                delta = np.abs(Sp2-Sp1)
                
                k = i
                
                if (delta<=self.eps):
                    
                    break
            
            plotter.plot(x,y,A)
            
            return (A,k)
        
        

class fd:
    
    def __init__(self,M,N,pot,cell,plotter):
        
        self.n = N
        
        self.m = M
        
        self.U = pot
        
        self.c = cell
        
        self.plotter = plotter
        
    def EX(self):
        
        ex = (self.U[:,1:] - self.U[:,:-1])/(2*self.c)
        
        x = np.linspace(0.,self.n*self.c,self.n-1)
        
        y = np.linspace(0.,self.m*self.c,self.m)
        
        plotter.plot(x,y,ex)
        
        return ex
    
    def EY(self):
        
        ey = (self.U[1:,:] - self.U[:-1,:])/(2*self.c)
        
        x = np.linspace(0.,self.n*self.c,self.n)
        
        y = np.linspace(0.,self.m*self.c,self.m-1)
        
        plotter.plot(x,y,ey)
        
        return ey
    
class poisson2:
    
    def __init__(self,M,N,cell,density,iterations,ledge,redge,up,down,nedge,Epsilon,plotter):
        
        self.n = N
        
        self.m = M
        
        self.c=cell
        
        self.dp=density
        
        self.ir=iterations
        
        self.le=ledge
        
        self.re=redge
        
        self.up=up
        
        self.dn=down
        
        self.eps=Epsilon
        
        self.plotter=plotter
        
        self.ne=nedge
        
        
    def Jacoby(self):
        
        x=np.linspace(0.,self.n*self.c,self.n)
        
        y=np.linspace(0.,self.m*self.c,self.m)
        
        A=np.zeros((self.m, self.n))
        
        A=self.ne
        
        A[0,:]=self.up
        
        A[-1,:]=self.dn
        
        A[:,0]=self.le
        
        A[:,-1]=self.re
        
        for i in range(self.ir):
            
            B=0.25*( A[0:-2,1:-1] + A[1:-1,0:-2] + A[1:-1,2:] + A[2:,1:-1])+np.pi*self.dp[1:-1,1:-1]*self.c**2
            A[1:-1,1:-1]=B
            A[15:-15,19:21]=100.
            A[15:-15,-21:-19]=-100.
        
        plotter.plot(x,y,A)
        
        return (A)


class densityofcharge: 
                
    def __init__(self,M,N,pot,cell,plotter):
        
        self.n = N
        
        self.m = M
        
        self.U = pot
        
        self.c = cell
        
        self.plotter = plotter
        
    def density(self):
        
        A = self.U
        
        d = (A[1:-1,1:-1] - 0.25*(A[0:-2,1:-1] + A[1:-1,0:-2] + A[1:-1,2:] + A[2:,1:-1]))/(np.pi*self.c**2)
        
        x = np.linspace(0.,self.n*self.c,self.n-2)
        
        y = np.linspace(0.,self.m*self.c,self.m-2)
        
        plotter.plot(x,y,d)
        
        return d


a = 100
b = 100

upm = np.zeros(b)

dnm = np.zeros(b)

lem = np.zeros(a)[:]=100.

rem = np.zeros(a)

mxp = np.zeros((a,b))

w = 1.0
            


plotter = VolPlotter('Gauss', 'x', 'y')    
        
solveGZ = poisson(a,b,1,mxp,5000,lem,rem,upm,dnm,0.01,plotter)

Z2=solveGZ.Gauss()

plotter.show()
            

plotter = VolPlotter('standart', 'x', 'y')

solveL = poisson(a,b,1,mxp,10000,lem,rem,upm,dnm,0.01,plotter)

Z0 = solveL.standart(100)

plotter.show()

            
plotter = VolPlotter('Jacoby', 'x', 'y')

solveL = poisson(a,b,1,mxp,10000,lem,rem,upm,dnm,0.01,plotter)

Z1 = solveL.Jacoby()

plotter.show()



upm = np.zeros(b)

dnm = np.zeros(b)

lem = np.zeros(a)[:] = 100.

rem = np.zeros(a)[:] = -100.

mxp = np.zeros((a,b))


plotter = VolPlotter('condensator', 'x', 'y')

solveL = poisson(a,b,1,mxp,10000,lem,rem,upm,dnm,0.01,plotter)

Z5 = solveL.Jacoby()

plotter.show()


upm = np.zeros(b)

dnm = np.zeros(b)

lem = np.zeros(a)

rem = np.zeros(a)

mxp = np.zeros((a,b))

mxp[15:-15,20] = 5.

mxp[15:-15,-20] = -5.


plotter = VolPlotter('poisson', 'x', 'y')

solveL = poisson(a,b,1,mxp,10000,lem,rem,upm,dnm,0.01,plotter)

Z4 = solveL.Jacoby()

plotter.show()


plotter = VolPlotter('EX', 'x', 'y')

field = fd(a,b,Z4,1,plotter)

ex = field.EX()

plotter.show()


plotter = VolPlotter('EY', 'x', 'y')

field = fd(a,b,Z4,1,plotter)

ey = field.EY()

plotter.show()


upm = np.zeros(b)

dnm = np.zeros(b)

lem = np.zeros(a)

rem = np.zeros(a)

mxp = np.zeros((a,b))

U1 = np.zeros((a,b))

U1[15:-15,19:21] = 100.

U1[15:-15,-21:-19] = -100.

plotter = VolPlotter('Capacity', 'x', 'y')

solveL = poisson2(a,b,1,mxp,10000,lem,rem,upm,dnm,U1,0.01,plotter)

Z6 = solveL.Jacoby()

plotter.show()

plotter = VolPlotter('Charge', 'x', 'y')

ch = densityofcharge(a,b,Z6,1,plotter)

fin = ch.density()

plotter.show()
