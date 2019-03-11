# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:52:17 2017

@author: ITA
"""

# -*- coding: utf-8 -*-
"""
This script determines the element that has the maximum value of Sigmaxx after
the printing has finished.

V4: READS ALSO STRESSES AT TWO FRAMES OF THE LAST STEP (INITIAL AND FINAL)

V5: FLAG TO CHOOSE IF BOTH FRAMES ARE SAVED
"""
#from part import *
#from material import *
#from section import *
#from assembly import *
#from step import *
#from interaction import *
#from load import *
#from mesh import *
#from optimization import *
#from job import *
#from sketch import *
#from visualization import *
#from connectorBehavior import *

#import math
#import numpy as np
from odbAccess import *
from abaqusConstants import *

import odbAccess

from multiprocessing import Pool
from functools import partial

import math
import numpy as np

import os, os.path

class SHAPE_FUNCTIONS:
    def __init__(self, DIM,NNODES):
        self.DIMENSION = DIM
        self.number_nodes = NNODES
        
    def natural_nodes(self, vecX, vecY, vecZ):
        self.X = vecX
        self.Y = vecY
        self.Z = vecZ
        
    def nodal_temperature(self,vec_TEMP):
        self.TEMP_NODAL = vec_TEMP
    
    def brick_element(self,r,s,t):
        oneO8 = 1.0/8.0
        one = 1.0
        
        N1 = oneO8 *(1-r)*(1-s)*(1-t)
        
        N2 = oneO8 *(1+r)*(1-s)*(1-t)
        
        N3 = oneO8 *(1+r)*(1+s)*(1-t)
        
        N4 = oneO8 *(1-r)*(1+s)*(1-t)
        
        N5 = oneO8 *(1-r)*(1-s)*(1+t)
        
        N6 = oneO8 *(1+r)*(1-s)*(1+t)
        
        N7 = oneO8 *(1+r)*(1+s)*(1+t)
        
        N8 = oneO8 *(1-r)*(1+s)*(1+t)
        
        vec_shape = np.zeros((1,8))   
        vec_shape[0] = [N1,N2,N3,N4,N5,N6,N7,N8]
        
        dN1dr = oneO8 *(-one)*(1-s)*(1-t)
        dN1ds = oneO8 *(1-r)*(-one)*(1-t)
        dN1dt = oneO8 *(1-r)*(1-s)*(-one)
        
        dN2dr = oneO8 *(one)*(1-s)*(1-t)
        dN2ds = oneO8 *(1+r)*(-one)*(1-t)
        dN2dt = oneO8 *(1+r)*(1-s)*(-one)
        
        dN3dr = oneO8 *(one)*(1+s)*(1-t)
        dN3ds = oneO8 *(1+r)*(one)*(1-t)
        dN3dt = oneO8 *(1+r)*(1+s)*(-one)
        
        dN4dr = oneO8 *(-one)*(1+s)*(1-t)
        dN4ds = oneO8 *(1-r)*(one)*(1-t)
        dN4dt = oneO8 *(1-r)*(1+s)*(-one)
        
        dN5dr = oneO8 *(-one)*(1-s)*(1+t)
        dN5ds = oneO8 *(1-r)*(-one)*(1+t)
        dN5dt = oneO8 *(1-r)*(1-s)*(one)
        
        dN6dr = oneO8 *(one)*(1-s)*(1+t)
        dN6ds = oneO8 *(1+r)*(-one)*(1+t)
        dN6dt = oneO8 *(1+r)*(1-s)*(one)
        
        dN7dr = oneO8 *(one)*(1+s)*(1+t)
        dN7ds = oneO8 *(1+r)*(one)*(1+t)
        dN7dt = oneO8 *(1+r)*(1+s)*(one)
        
        dN8dr = oneO8 *(-one)*(1+s)*(1+t)
        dN8ds = oneO8 *(1-r)*(one)*(1+t)
        dN8dt = oneO8 *(1-r)*(1+s)*(one)
        
        MAT_diff = np.zeros((8,3))
        
        MAT_diff[0] = [ dN1dr , dN1ds , dN1dt ]
        MAT_diff[1] = [ dN2dr , dN2ds , dN2dt ]
        MAT_diff[2] = [ dN3dr , dN3ds , dN3dt ]
        MAT_diff[3] = [ dN4dr , dN4ds , dN4dt ]
        MAT_diff[4] = [ dN5dr , dN5ds , dN5dt ]
        MAT_diff[5] = [ dN6dr , dN6ds , dN6dt ]
        MAT_diff[6] = [ dN7dr , dN7ds , dN7dt ]
        MAT_diff[7] = [ dN8dr , dN8ds , dN8dt ]

        self.shapefun = vec_shape
        self.shape_diff = MAT_diff
        
    def JACOBIAN(self):
#    NODE_COORD.X = x coordinates  
        if self.DIMENSION == 3:
            JACOBIAN = np.zeros((3,3))
            for i in range(0,3):
                somax = 0.0
                somay = 0.0
                somaz = 0.0
                for k in range(0,self.number_nodes):
                    aux = self.X[k]*self.shape_diff[k][i]
                    somax = somax + aux
                    
                    aux = self.Y[k]*self.shape_diff[k][i]
                    somay = somay + aux
                    
                    aux = self.Z[k]*self.shape_diff[k][i]
                    somaz = somaz + aux
                    
                JACOBIAN[i,0] = somax
                JACOBIAN[i,1] = somay
                JACOBIAN[i,2] = somaz
                
        self.detJACOBIAN = np.linalg.det(JACOBIAN)
        
        
    def TEMPERATURE_RST(self):
        TEMP_RST = 0.0
        for i in range(0,len(self.TEMP_NODAL)):
            TEMP_RST = TEMP_RST + self.TEMP_NODAL[i]*self.shapefun[0][i]
        self.TEMP_RST = TEMP_RST
        
    def FUNCTION_INTEGRATED_RST(self,r,s,t):
        
        self.brick_element(r,s,t) #changes shapefun and diff shapefun
        
        self.JACOBIAN() #calculates Jacobian at new r,s,t
        
        self.TEMPERATURE_RST()
        
        f = self.TEMP_RST*abs(self.detJACOBIAN)
        
        return f
#------------------------------------------------------------------------------
class GAUSS_POINTS:
    def __init__(self, npoints):
        self.points = npoints
        
        #W = weight
        #P = point
        if npoints == 2:
            self.W = [1.0,1.0]
            
            root3 = math.sqrt(3)
            self.P = [-1/root3, 1/root3]
            
        elif npoints == 1:
            self.W = [2.0]
            self.P = [0.0]
            
        elif npoints == 3:
            root3 = math.sqrt(3)
            root5 = math.sqrt(5)
            
            fiveOnine = 5./9.
            
            self.W = [fiveOnine, 8./9., fiveOnine]
            self.P = [-root3/root5,0.0, root3/root5]
            
#==============================================================================
# QUAD GAUSS 2D OR 3D
#==============================================================================
def gauss_quad(DIM,func_name):
    Nx = 3
    Ny = 3
    Nz = 3
    
    X = GAUSS_POINTS(Nx)
    Y = GAUSS_POINTS(Ny)
    Z = GAUSS_POINTS(Nz)
    
    sum_gauss = 0.0
    
    if DIM == 2:
        for j in range(0,Ny):
            
            p_x2 = Y.P[j]
            w_x2 = Y.W[j]
            
            for i in range(0,Nx):
                p_x1 = X.P[i]
                w_x1 = X.W[i]
                
                point = (p_x1,p_x2)
                
                f = func_name(point)
                
                sum_gauss = sum_gauss + w_x1*f
                
            sum_gauss = sum_gauss * w_x2
            
    elif DIM == 3:
        for k in range(0,Nz):
            p_x3 = Z.P[k]
            w_x3 = Z.W[k]
            
            aux_gauss_j = 0.0
            for j in range(0,Ny):
                p_x2 = Y.P[j]
                w_x2 = Y.W[j]
                
                aux_gauss_i = 0.0
                for i in range(0,Nx):
                    p_x1 = X.P[i]
                    w_x1 = X.W[i]
                
                    point = (p_x1,p_x2,p_x3)
                    
                    f = func_name(p_x1,p_x2,p_x3)
                    
                    aux_gauss_i = aux_gauss_i + w_x1*f
                    
                aux_gauss_j = aux_gauss_j + aux_gauss_i * w_x2
            sum_gauss = sum_gauss + aux_gauss_j* w_x3
    
          
    return sum_gauss
    
#------------------------------------------------------------------------------    
#==============================================================================
# WRITING VECTORS OF NODAL X, Y AND Z
#==============================================================================
def rewrite_vec_pos(newnodes,ELE_TYPE):
    vecX = list()
    vecY = list()
    vecZ = list()
    for i in range(0,ELE_TYPE):
        node_i = newnodes[i]
        
        vecX.append(node_i[0])
        vecY.append(node_i[1])
        vecZ.append(node_i[2])
        
    return vecX,vecY,vecZ
    
#==============================================================================
# CALCULATE DISTANCE BETWEEN 2 POINTS
#==============================================================================
def distance(coord1,coord2):
    aux = 0.0
    for i in range(0,len(coord1)):
        aux_i = (coord1[i] - coord2[i])**2
        aux = aux + aux_i
    
    dist = math.sqrt(aux)
    return dist
    
#==============================================================================
# CALCULATING NEW COORDINATES FOR EACH ELEMENT
#==============================================================================
def ELEMENT_DATA(Instance,R_table,table_nodes,table_CONEC,ele_number,ELE_TYPE):
    #Instance = number of the instance

    translate = R_table[Instance - 1] #translation vector

    NODES = table_CONEC[ele_number - 1] #node number of the current element
    
    new_coordinates = list() #coordinates of each node of the current element
    
    for i in range(0,ELE_TYPE):
        
        node_i = int(NODES[i])
        
        coord_i = table_nodes[node_i - 1] #coordinate of node _i
        
        aux_coord = list()
        for j in range(0,3):
            aux = float(coord_i[j]) + translate[j]
            
            aux_coord.append(aux)
            
        new_coordinates.append(aux_coord)
    
    lx = distance(new_coordinates[1],new_coordinates[2])
    ly = distance(new_coordinates[1],new_coordinates[0])
    lz = distance(new_coordinates[0],new_coordinates[4])
    
    Vol = lx * ly * lz #element volume
    
    return new_coordinates, Vol
    
#==============================================================================
# READING INP FILE
#==============================================================================
def read_inp(name_file_inp):   
    file_read = open(name_file_inp, 'r');#READS THE INP FILE
    
    ELE_TYPE = 8 #NUMBER OF NODES PER ELEMENT
    #JUMPING LINES
    for i in range(0,9):
        file_read.readline()
    
    #READING NODES -----------------------------------------------------------
    aux_stop = "t";
    
    table_nodes = list()
    while aux_stop != "*":
        node_aux = file_read.readline()
        aux_stop = node_aux[0]
        node_aux = node_aux.replace(",", "")
        node_aux = node_aux.split();
        table_nodes.append(node_aux[1:4])
    
    del table_nodes[-1] #DELETE LAST ELEMENT
    NUMBER_NODES = len(table_nodes)
    
    #------------------------------------------------------------------------------
    #READING ELEMENTS--------------------------------------------------------------
    aux_stop = "t";
    
    table_CONEC = list()
    while aux_stop != "*":
        ele_aux = file_read.readline()
        aux_stop = ele_aux[0]
        ele_aux = ele_aux.replace(",", "")
        ele_aux = ele_aux.split();
        while len(ele_aux) < (ELE_TYPE + 1):
            ele_aux2 = file_read.readline()
            ele_aux2 = ele_aux2.replace(",", "")
            ele_aux2 = ele_aux2.split();
            ele_aux = ele_aux + ele_aux2
          
        table_CONEC.append(ele_aux[1:ELE_TYPE+1])
        
    
    del table_CONEC[-1] #DELETE LAST ELEMENT
    NUMBER_ELE = len(table_CONEC) #ELEMENTS PER INSTANCE
    
    #------------------------------------------------------------------------------
    #READING THE VECTOR R OF ASSEMBLY
    aux_stop = "t";
    
    while aux_stop != "*Instance, name=I-1":
        aux_stop = file_read.readline()
        aux_stop = aux_stop[0:19]
     
    aux_stop = "*I"
    R_table = list()
    LAYER_INSTANCE = list() #LIST THAT SAYS WHICH LAYER EACH INSTANCE IS AT
    LAYER_INSTANCE.append(1) #FIRST INSTANCE = LAYER 1
    aux_instance_layer = 0.0
    layer_cont = 1
    while aux_stop == "*I":
        file_read.readline()
        file_read.readline()
        
        aux_stop = file_read.readline()
        aux_stop = aux_stop[0:2]
        
        coord_R = file_read.readline()
        coord_R = coord_R.replace(",", "")
        coord_R = coord_R.split();
    
        for i in range(0,3):
            coord_R[i] = float(coord_R[i])
        
        #CHECKING THE LAYER INSTANCES
        if float(coord_R[2]) > aux_instance_layer:
           aux_instance_layer = float(coord_R[2])
           layer_cont = layer_cont + 1
           
        LAYER_INSTANCE.append(layer_cont)
        R_table.append(coord_R)
    
    del R_table[-1] #DELETE LAST ELEMENT
    del LAYER_INSTANCE[-1] #DELETE LAST ELEMENT
    NUMBER_LAYER = max(LAYER_INSTANCE) #NUMBER OF LAYERS
    R_table.insert(0,[0.0, 0.0, 0.0]) #FIRST INTANCE R = [0 0 0]
    NUMBER_INSTANCE = len(R_table)
    NUMBER_ELE = len(table_CONEC) #ELEMENTS PER INSTANCE
    file_read.close()
    
    return R_table, table_nodes, table_CONEC

#==============================================================================
# READ ODB
#==============================================================================
def read_odb(i,name_old,num_steps,num_instances,print_pattern,ELE_TYPE,R_table,table_nodes,table_CONEC,BRICK):
    
    myOdb = odbAccess.openOdb(path=name_old, readOnly = True)
     
    ALLInstances = myOdb.rootAssembly.instances
       
        
    name_new = name_old.replace('.odb','_%d.txt' %(i+1))
    
    report_name = 'report-%d.txt' %(i+1)
    
    file_report = open(report_name,'w')
    file_report.close()
    #RESTART:
    aux_restart = 0 #DOES NOT RESTART
    
    if os.path.isfile(name_new): #it file exists
        file_new = open(name_new,'r')
        
        cont_line = 0
        
        for line in file_new:
            cont_line = cont_line + 1
            
        file_new.close()
        if cont_line < 14:
            aux_restart = 1
    
    if not os.path.isfile(name_new) or aux_restart == 1: #it file does not exists or has not been written entirely
        file_new = open(name_new,'w')
        
        total_time = 0.0
        
        
        if i == num_steps - 1: #last step
            step_name = 'Step-FINAL-THERMAL-STEP'
            num_instance_now = num_instances
        else:     
            step_name =  'Step-%d' %(i+1)
            num_instance_now = i + 1
        
        step_now = myOdb.steps[step_name]
        num_frames = len(step_now.frames)
        
        file_new.write("--------------------------------------------------\n")
        file_new.write("%s\n" %step_name)
        
        for j in range(0,num_frames):
            frame_now = step_now.frames[j]
            
            time_step = frame_now.frameValue
            total_time = total_time + time_step
            
            TEMP_aux = frame_now.fieldOutputs['TEMP']  
            
            file_new.write("FRAME = %d ----------------------------------------\n" %j)
            file_new.write("TIME [s] = %f\n" %total_time)
    
            cont_ele = 0 #global elements
            
            Vol_total = 0.0 #volume total is zeroed in each frame
        
            Int_temp_total = 0.0 #integration sum variable is zeroed in each frame
            
            file_new_frame = name_new.replace('.txt','') + 'F%d.txt' %(j)
            
            for k in range(1,num_instance_now+1):
                
                intance_number = print_pattern[k-1]
                
                file_name_new_I = file_new_frame.replace('.txt','') + '_I%d.txt'  %(intance_number)
                file_new_I = open(file_name_new_I,'w')
                
                instanceName = 'I-%d' %intance_number
                
                myInstance = ALLInstances[instanceName]
                
                numElements = len(myInstance.elements)
    
                Vol_total = 0.0 #volume total is zeroed in each frame
        
                Int_temp_total = 0.0 #integration sum variable is zeroed in each frame
                
                for el in range(0,numElements):
                    #Isolate current and previous element's stress field
                    
                    #THOSE RESULTS ARE NO AVERAGED
                    #POSITION = INTEGRATION_POINT/ELEMENT_NODAL/CENTROID
                    region_aux = myInstance.elements[el]
                    
                    TEMP_NODAL = TEMP_aux.getSubset(
                    region=region_aux,position=ELEMENT_NODAL,elementType='DC3D8').values
                    
                    cont_ele = cont_ele +1
                       
                    local_ele = el+1
                    
                    TEMP_NODE_ip = list()
                    
                    for ip in range(0,ELE_TYPE): #node loop
                        
                        TEMP_NODE_ip.append(TEMP_NODAL[ip].data)
                    
                    
                    (new_coordinates, Vol) = ELEMENT_DATA(intance_number,R_table,table_nodes,table_CONEC,local_ele,ELE_TYPE)    
                    
                    Vol_total = Vol_total + Vol
                    
                    #Post-Processing ----------------------------------------------
                    (vecX,vecY,vecZ) = rewrite_vec_pos(new_coordinates,ELE_TYPE)
                    
                    BRICK.natural_nodes(vecX, vecY, vecZ)
                    
                    BRICK.nodal_temperature(TEMP_NODE_ip)
                    
    #                print BRICK.nodal_temperature
                    
                    #Integrating temperature distribution within the element
                    Iaux = gauss_quad(BRICK.DIMENSION,BRICK.FUNCTION_INTEGRATED_RST)
                    Int_temp_total = Int_temp_total + Iaux
                    
                    
                file_new_I.write("Integral Volume\n")
                file_new_I.write('%f %E\n'%(Int_temp_total/Vol_total,Vol_total))
                file_new_I.close()

        file_new.close()
    myOdb.close()
#------------------------------------------------------------------------------                

if __name__ == '__main__':   
    
    ELE_TYPE = 8
    pool = Pool(processes=20)
    #Odb file in the work directory:
    odb_name = 'job_sim1_mesh_2_2_2.odb' 
    
    myOdb = odbAccess.openOdb(path=odb_name, readOnly = True)
    #--------------------------------------------------------------------------
    
    ALLInstances = myOdb.rootAssembly.instances
    num_instances = len(ALLInstances)
    
    mysteps = myOdb.steps
    num_steps = len(mysteps)
#    myOdb.close()
    
    #==========================================================================
    # Reading report file
    #========================================================================== 
    report_file_name = 'REPORT_sim1_mesh_2_2_2.txt'
    file_report = open(report_file_name,'r')
    
    print_pattern = list()
    
    for i in range(0,24):
        file_report.readline() #jumping lines
    for i in range(0,num_instances):
        print_pattern.append(int(file_report.readline()))
         
    file_report.close()  
    
    
    #--------------------------------------------------------------------------
    
    #==============================================================================
    #READING INP FILE
    #==============================================================================
    name_file_inp = 'job_sim1_mesh_2_2_2.inp' #INP FILE OF THE ANALYSIS(MUST CONTAIN THE ASSEMBLY)
    R_table, table_nodes, table_CONEC = read_inp(name_file_inp)
    

    # -------------------------------------------------------------------------
    # POST-PROCESSING
    BRICK = SHAPE_FUNCTIONS(3,8)
    
    #Parallel processing
    inputs = range(num_steps)
#    inputs = range(1)
    aux = pool.map(partial(read_odb, name_old=odb_name, num_steps = num_steps,num_instances=num_instances, 
                           print_pattern=print_pattern,ELE_TYPE=ELE_TYPE,R_table=R_table,table_nodes=table_nodes,table_CONEC=table_CONEC,BRICK=BRICK), inputs)
    
    myOdb.close()