import numpy as np
from matplotlib import lines as line
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.lines import Line2D

class plotter: 
	
    def __init__(self): 
        
        self.fig = plt.figure()
        
        self.ax = self.fig.add_subplot(111)
        
    def plot(self, x, y): 
        
        return plt.plot(x, y,)

    def show(self):
        
        plt.show()
        
        
        
class st:
    

    def __init__(self, initial_velocity, initial_angle, grav_field, mass, plott):
        
        self.vi = initial_velocity
        
        self.m = mass
        
        self.phi = initial_angle
        
        self.g = grav_field
        
        self.plotr = plott
    
    def bias_x(self,t):
        
        return self.vi*np.cos(self.phi)*t
    
    
    def bias_y(self,t):
        
        return self.vi*np.sin(self.phi)*t-self.g*t**2/2
    
    
    def track(self,t1, max_t):
        
        t = np.linspace(0,t1,max_t)
        
        x = self.bias_x(t)
        
        y = self.bias_y(t)
        
        self.plotr.plot(x,y)
        
        
gant = st(10, np.pi/8, 9.82, 1.0, plotter())

gant.track(10,10)




class gantel(st):
    
    def __init__(self, initial_velocity, initial_angle, grav_field, mass, plott, radius, frequancy):
        
        self.r = radius
        
        self.fq = frequancy
        
        st.__init__(self, initial_velocity, initial_angle, grav_field, mass, plott)
        
    
    def trajectory(self, t1, max_t):
        
         t=np.linspace(0,t1, max_t)
         
         x1 = self.bias_x(t )+ self.r*np.cos(self.fq*t)
         
         y1 = self.bias_y(t) + self.r*np.sin(self.fq*t)
         
         self.plotr.plot(x1,y1)
         
         x2 = self.bias_x(t) + self.r*np.cos(self.fq*t-np.pi)
         
         y2 = self.bias_y(t) + self.r*np.sin(self.fq*t-np.pi)
         
        
         self.plotr.plot(x2,y2)
         
         for i in range(0,max_t):
             
           l = line.Line2D([x1[i],x2[i]], [y1[i],y2[i]], color='red')
           
           self.plotr.ax.add_line(l)
        
             
gantel1 = gantel(10, np.pi/8, 9.82, 1.0, plotter(),5, -np.pi/6)

gantel1.trajectory(10,100)




class dip(gantel):
    
    def __init__(self,initial_velocity, initial_angle, grav_field, mass, plott, radius, frequancy, charge, elect_field):
        
        self.om = frequancy
        
        self.q = charge
        
        self.E = elect_field
        
        self.mass = mass
        
        
        self.radius = radius
        
        st.__init__(self, initial_velocity, initial_angle, grav_field, mass, plott)
        
        gantel.__init__(self,initial_velocity, initial_angle, grav_field, mass, plott, radius, frequancy)
        
        
    def trdip(self,t1, max_t):
        
        t = np.linspace(0,t1,max_t)
        
        def equation(y,t):
            
         fq,om = y
         
         return [om, (self.mass*self.radius)/(self.q*self.E)*np.sin(fq)]
     
        chastota=sp.integrate.odeint(equation,[0,self.om],t)[:,0]
        
        t=np.linspace(0,t1,max_t)
        
        x1 = self.bias_x(t) + self.r*np.cos(chastota)
        
        y1 = self.bias_y(t) + self.r*np.sin(chastota)
        
        self.plotr.plot(x1,y1)
        
        x2 = self.bias_x(t) + self.r*np.cos(chastota-np.pi)
        
        y2 = self.bias_y(t) + self.r*np.sin(chastota-np.pi)
        
        self.plotr.plot(x2,y2)
        
        for i in range(0,max_t):
            
           l = line.Line2D([x1[i],x2[i]], [y1[i],y2[i]], color='blue')
           
           self.plotr.ax.add_line(l)
        
d=dip(10, np.pi/4, 9.82, 1.0, plotter(),5, -np.pi/6,1.0,1.0)

d.trdip(10,10)


