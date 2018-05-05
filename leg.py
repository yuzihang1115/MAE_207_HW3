import odrive.core
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# For symbolic processing
import sympy
from sympy import symbols
from sympy import sin, cos, asin, acos, pi, atan,sqrt
from sympy.utilities.lambdify import lambdify
from sympy import Matrix


#%matplotlib qt5 #opens new window for plotting things on the same figure

class Leg:
    """
    This is our first class in class :)

    We will define a leg class to interface with the leg and standardize 
    the kind of operations we want to perform

    """
    global l1, l2, l_base, theta0_sym, theta1_sym, alpha0_sym, alpha1_sym, encoder2angle
    #### Variables outside the init function are constants of the class
    # leg geometry
    l1 = 7.1  # NEED TO UPDATE units of cm
    l2 = 14.1  # NEED TO UPDATE units of cm
    l_base = 7.6  # NEED TO UPDATE units of cm
    theta0_sym, theta1_sym, alpha0_sym, alpha1_sym, = symbols(
            'theta0_sym theta1_sym alpha0_sym alpha1_sym' , real=True)

    # motor controller parameters
    encoder2angle = 2048 * 4

    ### Methods
    # Classes are initiated with a constructor that can take in initial parameters. At
    # a minimum it takes in a copy of itself (python... weird). The constructor
    # is one place we can define and initialize class variables

    def __init__(self, simulate = False):
        """
        This is the constructor for the leg class. Whenever you make a new leg
        this code will be called. We can optionally make a leg that will simulate
        the computations without needing to be connected to the ODrive
        """

        self.simulate = simulate #simulate            

        # make the option to code without having the odrive connected
        if self.simulate == False:
            self.drv = self.connect_to_controller()
            self.m0 = self.drv.motor0  # easier handles to the motor commands
            self.m1 = self.drv.motor1

            # current positions
            m0_pos, m1_pos = self.get_joint_pos()
            self.joint_0_pos = m0_pos
            self.joint_1_pos = m1_pos

        else:
            self.drv = None
            self.joint_0_pos = 1.57
            self.joint_1_pos = 1.57

        # home angles
        self.joint_0_home = 0
        self.joint_1_home = 0
        self.home=[0,0]


        # We will compute the jacobian and inverse just once in the class initialization.
        # This will be done symbolically so that we can use the inverse without having
        # to recompute it every time
        self.J = self.compute_jacobian()

    def connect_to_controller(self):
        """
        Connects to the motor controller
        """
        drv = odrive.core.find_any(consider_usb=True, consider_serial=False)

        if drv is None:
            print('No controller found')
        else:
            print('Connected!')
        return drv

    ###
    ### Motion functions
    ###
    def get_joint_pos(self):
        """
        Get the current joint positions and store them in self.joint_0_pos and self.joint_1_pos in degrees.
        Also, return these positions using the return statement to terminate the function
        """
        # if simulating exit function
        if self.simulate == True:
            return (self.joint_0_pos, self.joint_1_pos)
        else:
            
            self.joint_0_pos = self.m0.encoder.pll_pos/encoder2angle*(2*pi) +pi/2 #degree conversion
            self.joint_1_pos = self.m1.encoder.pll_pos/encoder2angle*(2*pi) +pi/2


        return (self.joint_0_pos, self.joint_1_pos)

    def set_home(self):
        """
        This function updates the home locations of the motors so that 
        all move commands we execute are relative to this location. 
        """
        # if simulating exit function
        if self.simulate == True:
            return
        else: # Your code here
            self.home = (self.m0.encoder.pll_pos/encoder2angle*(2*pi)+pi/2, self.m1.encoder.pll_pos/encoder2angle*(2*pi)+pi/2) #angle


    def set_joint_pos(self, theta0, theta1, vel0, vel1, curr0, curr1):
        """
        Set the joint positions in units of deg, and with respect to the joint homes.
        We have the option of passing the velocity and current feedforward terms.
        """
        # if simulating exit function
        if self.simulate == True:
            self.joint_0_pos = theta0
            self.joint_1_pos = theta1
        else: # Your code here
            self.get_joint_pos()
            self.m0.set_pos_setpoint((theta0-self.home[0])*encoder2angle/(2*pi)-pi/2 ,vel0,curr0) #encoder value = angle*(-4000)/pi determiend in first homework
            self.m1.set_pos_setpoint((theta1-self.home[1])*encoder2angle/(2*pi)-pi/2 ,vel1,curr1)
            
            

    def move_home(self):
        """
        Move the motors to the home position
        """
        # if simulating exit function
        if self.simulate == True:
            return
        else: # Your code here
            self.m0.set_pos_setpoint(self.home[0])
            self.m1.set_pos_setpoint(self.home[1])
        

    def set_foot_pos(self, x, y):
        """
        Move the foot to position x, y. This function will call the inverse kinematics 
        solver and then call set_joint_pos with the appropriate angles
        """
        # if simulating exit function
        if self.simulate == True:
            (theta_0, theta_1) = self.IKK(x,y)
            return (theta_0, theta_1)
        else:
            (theta_0, theta_1) = self.IKK(x,y)
            self.set_joint_pos(theta_0,theta_1) 
       

    def move_trajectory(self, tt, xx, yy):
        """
        Move the foot over a cyclic trajectory to positions xx, yy in time tt. 
        This will repeatedly call the set_foot_pos function to the new foot 
        location specified by the trajectory.
        """
        # if simulating exit function        

        if self.simulate == True:
            tra_theta0, tra_theta1 = [], []
            tra_alpha0, tra_alpha1 = [], []
            
            for i in range(tt):
                # Computer theta's for trajectory and store them
                theta_0, theta_1 = self.set_foot_pos(xx[i],yy[i])
                tra_theta0.append(theta_0)
                tra_theta1.append(theta_1)

                # Computer alpha's for trajectory and store them
                alpha_0, alpha_1 = self.compute_internal_angles(theta_0, theta_1)
                tra_alpha0.append(alpha_0)
                tra_alpha1.append(alpha_1)

            return (tra_theta0, tra_theta1, tra_alpha0, tra_alpha1)    
                
        else:
            for i in range(tt):
                self.set_foot_pos(xx[i],yy[i])
                          
      
    ###
    ### Leg geometry functions
    ###
    
    def compute_internal_angles(self, theta_0, theta_1):
        """
        Return the internal angles of the robot leg
        from the current motor angles
        """

        D = sqrt(l_base ** 2 + l1 ** 2 - 2 * l1 * l_base * cos(math.pi - theta_0))
        Beta = asin((l1 / D) * sin(math.pi - theta_0))
        Q1 = theta_1 - Beta

        A = 2 * l2 * (l1 * cos(Q1) - D)
        B = 2 * l1 * l2 * sin(Q1)
        C = 2 * D * l1 * cos(Q1) - (l1 ** 2) - (D ** 2)
        Q2 = (atan(B / A) - acos(C / sqrt(A * A + B * B))) + math.pi  # +or-

        alpha_1 = Q2 + Beta
        alpha_0 = math.pi - asin((l1 * sin(theta_1) - l1 * sin(theta_0) + l2 * sin(alpha_1)) / l2)

        return (alpha_0, alpha_1)

 
    def compute_jacobian(self):
        """
        This function implements the symbolic solution to the Jacobian.
        """

        # initiate the symbolic variables
        theta0_sym, theta1_sym, alpha0_sym, alpha1_sym = symbols(
            'theta0_sym theta1_sym alpha0_sym alpha1_sym', real=True)


        [alpha0_sym, alpha1_sym]=self.compute_internal_angles(theta0_sym, theta1_sym)
        x=(l_base/2)+l1*cos(theta0_sym)+l2*cos(alpha0_sym)
        y=l1*sin(theta0_sym)+l2*sin(alpha0_sym)
        FK=Matrix([[x],[y]])
        J=FK.jacobian([theta0_sym,theta1_sym])

        return J

    
    def IKK(self,x_t,y_t):


        theta0_sym, theta1_sym, alpha0_sym, alpha1_sym = symbols(
            'theta0_sym theta1_sym alpha0_sym alpha1_sym', real=True)

        J_current = lambdify((theta0_sym, theta1_sym), self.J)

        (theta_0,theta_1) = self.get_joint_pos()
     
        theta_current=np.array([[theta_0],[theta_1]])

        # solution parameters
        beta = 0.1    # step size
        epsilon = 0.1     # stop error
        for kk in range(1000):

            (alpha_0,alpha_1) = self.compute_internal_angles(theta_current[0],theta_current[1])

            J_temp=J_current(theta_0,theta_1)

            J_inv = np.linalg.pinv(J_temp)

            x=(l_base/2)+l1*cos(theta_current[0])+l2*cos(alpha_0)
            y=l1*sin(theta_current[0])+l2*sin(alpha_0)

            x_current = np.array([x,y])
            x_target = np.array([x_t,y_t])

            x_delta = (x_target - x_current)

            error = sympy.N(Matrix([x_t-x, y_t-y]))
            if error.norm()< 0.1:
                break
            x_delta_t = np.array([x_delta]).T
            theta_current = beta*np.dot(J_inv, x_delta_t) + theta_current

            theta_current = np.array(theta_current).astype(np.float64)
            self.set_joint_pos(theta_0,theta_1)
        return (theta_current[0],theta_current[1])


    def my_draw_leg(self, theta1, theta2, alpha1, alpha2, ax=False):
        """
        This function takes in the four angles of the leg and draws
        the configuration
        """

        link1, link2, width = l1, l2, l_base
        cor_range = l1 + l2

        if ax == False:
            ax = plt.gca()
            ax.cla()

        ax.plot(-width / 2, 0, 'ok')
        ax.plot(width / 2, 0, 'ok')

        ax.plot([-width / 2, 0], [0, 0], 'k')
        ax.plot([width / 2, 0], [0, 0], 'k')

        ax.plot(-width / 2 + np.array([0, link1 * cos(theta1)]), [0, link1 * sin(theta1)], 'k')
        ax.plot(width / 2 + np.array([0, link1 * cos(theta2)]), [0, link1 * sin(theta2)], 'k')

        ax.plot(-width / 2 + link1 * cos(theta1) + np.array([0, link2 * cos(alpha1)]), \
                link1 * sin(theta1) + np.array([0, link2 * sin(alpha1)]), 'k');
        ax.plot(width / 2 + link1 * cos(theta2) + np.array([0, link2 * cos(alpha2)]), \
                np.array(link1 * sin(theta2) + np.array([0, link2 * sin(alpha2)])), 'k');

        curr_x = float(width / 2 + link1 * cos(theta2) + link2 * cos(alpha2))
        curr_y = float(np.array(link1 * sin(theta2) + link2 * sin(alpha2)))

        ax.plot(curr_x, curr_y, 'ro');

        ax.axis([-cor_range, cor_range, -l1, cor_range])
        ax.invert_yaxis()