# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:58:36 2023

@author: xiety
"""

#注意本脚本只适用于不带缺陷的完美晶格体系
#仅仅用于找出并且删Ti周围的O原子

from pymatgen.core.composition import Composition
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
import random
import os
import sys


show_vac=0  #1 for use XX element to show vac position
##########You Need To Change Below Code In A Diff System##########
dir_files="C:\\Users\\xiety\\Desktop\\switch-LiNi\\Doping\\Doping_result\\doping_after_relax\\"
filename_dop="LiNiO2_331_NCM_513_Ti_num_1_0.vasp"
file_dop=dir_files+filename_dop

filename_orig="LiNiO2_331_NCM_513.vasp"
file_orig=dir_files+filename_orig


doping_ele="XX"   #doping atom element
doped_ele="O"     #doped atom element in original cell

sample_number=6 #this is for monte-carlo sample time

doping_num=1 #doping atom number

tolarence_dis=1e-5 #tolarence distance to judge two atom in one kind

#basic symmetric change

O_neighbor=Composition("O")
Li_neighbor=Composition("Li")
TM_ele=['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn']
TM_neighbor=[Composition(x) for x in TM_ele]

structure = Poscar.from_file(file_dop).structure
# create VoronoiNN object

ele_lst=[str(i.species)[:-1] for i in structure]
vec_a=structure.lattice.matrix[0]
vec_b=structure.lattice.matrix[1]
vec_c=structure.lattice.matrix[2]
A_vec = np.vstack([vec_a, vec_b, vec_c]).T
A_inv = np.linalg.inv(A_vec)
pbc_cell_333=[-1,0,1]


#这里的center_atom_number是vesta里面的原子标号，而不是index（具体是index+1）
def neighbor_get(center_atom_number,surrounding_type):
    center_atom_index = center_atom_number - 1    
    ele_info_neighbor=[]
    for i in range(len(structure)):
        dis_i=[]
        for a in pbc_cell_333: #contains the 3x3x3 supercell atoms and only get the minimal distance
            for b in pbc_cell_333:
                for c in pbc_cell_333:
                    site_i=np.array(structure.sites[i].coords)+vec_a*a+vec_b*b+vec_c*c
                    #calculate the distance between the neighbors and the center to select the nearest neighbors
                    dis_i.append(np.linalg.norm(site_i - np.array(structure.sites[center_atom_index].coords)))
        dis=min(dis_i)
        if dis!=0:#exclude self
            ele_info_neighbor.append([structure.sites[i].species,(i+1),structure.sites[i].coords,dis])
    
    ele_info_neighbor=sorted(ele_info_neighbor,key=lambda x:x[-1])#list sorted by distance
    neighbor_O_tot=[]
    neighbor_Li_tot=[]
    neighbor_TM_tot=[]
    
    for i in ele_info_neighbor:
        if i[0]==O_neighbor:
            neighbor_O_tot.append(i)
        elif i[0]==Li_neighbor:
            neighbor_Li_tot.append(i)    
        elif i[0] in TM_neighbor:
            neighbor_TM_tot.append(i)
    
    neighbor_O_sel=neighbor_O_tot[:6]
    neighbor_Li_sel=neighbor_Li_tot[:6]
    neighbor_TM_sel=neighbor_TM_tot[:6]
    neighbor_tot_sel=ele_info_neighbor[:6]
    #print("****************Selected Atom********************")
    #print(structure.sites[center_atom_index].species,(center_atom_index+1),find_cluster(center_atom_index+1),structure.sites[center_atom_index].coords)
    #print("*************************************************")
    atom_environment={}
    surr_atom=[]
    
    if surrounding_type=="judge":
        found_Ti=0
        Ti_num_count=0
        for i in neighbor_tot_sel:
            if str(i[0])[:-1]=="Ti":
                found_Ti=1
                Ti_num_count+=1
        if found_Ti:
            return True,Ti_num_count
        else:
            return False,Ti_num_count
    elif surrounding_type=="TM":
        for i in neighbor_TM_sel:
            surr_atom.append(f'{str(i[0])[:-1]}-{i[-1]:.6f}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    elif surrounding_type=="Li":
        for i in neighbor_Li_sel:
            surr_atom.append(f'{str(i[0])[:-1]}-{i[-1]:.6f}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    elif surrounding_type=="O":
        for i in neighbor_O_sel:
            surr_atom.append(f'{str(i[0])[:-1]}-{i[-1]:.6f}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    elif surrounding_type=="tot":
        for i in neighbor_tot_sel:
            surr_atom.append(f'{str(i[0])[:-1]}-{i[-1]:.6f}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    return surr_atom
#判断方法：找到最近邻的原子，如果最近邻三种原子的类型，距离啥都一样，姑且认为是对称结构。
surr_doped_atom={}
doped_ele_lst=[]
ele_o_index=[]
Ti_count=[]
for i in structure:    
    if str(i.species)[:-1]=="O":
        ele_o_index.append(structure.index(i))
        include_Ti,Ti_num=neighbor_get(structure.index(i)+1,"judge")
        if not include_Ti:
            continue
        else:
            Ti_count.append([f"{str(i.species)[:-1]}-{structure.index(i)+1}",Ti_num])
            doped_ele_lst.append(structure.index(i)+1)
            surr_doped_atom[f"{str(i.species)[:-1]}-{structure.index(i)+1}"]=[]
            surr_doped_atom[f"{str(i.species)[:-1]}-{structure.index(i)+1}"].append(neighbor_get(structure.index(i)+1,"tot"))
    

##########You Need To Change Above Code In A Diff System##########
##################################################################
##################################################################    
print(surr_doped_atom)

def judge_same_env(env1,env2):
    for i,v in enumerate(env1):
        number_lst1=[i.split("-")[0] for i in env1[i]]
        number_lst2=[i.split("-")[0] for i in env2[i]]
        dis_lst1=[float(i.split("-")[1]) for i in env1[i]]
        dis_lst2=[float(i.split("-")[1]) for i in env2[i]]
        #print(number_lst1,number_lst2)
        #print(dis_lst1,dis_lst2)
        for j,w in enumerate(number_lst1):
            if w!=number_lst2[j]:
                return False
        for k,x in enumerate(dis_lst1):
            if (x-dis_lst2[k]>tolarence_dis):
                return False
    return True
#aa=judge_same_env(surr_doped_atom["Li-24"],surr_doped_atom["Li-35"])
#print(aa)

srs_raw=[int(i.split("-")[1]) for i in [key for key,v in surr_doped_atom.items()]]
print("Absolute number",srs_raw)
srs=[i-min(ele_o_index) for i in srs_raw]
print("Relative number",srs)

Ti_count=sorted(Ti_count,key=lambda x:x[-1])
print(Ti_count)

keys_atom = list(surr_doped_atom.keys())

print("***The atom list below are equivalent:**")
print("**********************************")
for i in range(len(keys_atom)):
    for j in range(i+1, len(keys_atom)):
        
        result_judge=judge_same_env(surr_doped_atom[keys_atom[i]],surr_doped_atom[keys_atom[j]])
        #print(result_judge)
        if not result_judge:
            pass
        else:   
            srs[j]=srs[i]
            print(keys_atom[i],keys_atom[j])
print("**********************************")

tot_x_num=max(ele_o_index)-min(ele_o_index)+1   #X-site atom number
#print(tot_x_num,len(ele_o_index))
if doping_num> tot_x_num:
    raise ValueError(f"The total number of {doped_ele} is {tot_x_num} and the {doping_num} is too large  !")
srs_st=sorted(list(set(srs)))

#print(srs)
srs_st_index=[i-1 for i in srs_st]
srs_index=[i-1 for i in srs]
#monte-carlo part

def generate_sequence():
    seq = [1]*tot_x_num
    if doping_num<=len(srs_st):
        indices_to_replace = random.sample([i for i in srs_st_index], doping_num)
    else:
        indices_to_replace = random.sample(range(len(srs)), doping_num)
    for i in indices_to_replace:
        seq[i] = 0
    return seq

#a=generate_sequence()
#index_0=[i+1 for i,x in enumerate(a) if x==0]
# print(a,index_0)


str_list=[]
#str_list.append(generate_sequence())
print("Replace following")
dupli_cout=0
for i in range(sample_number):
    #print(i)
    c=generate_sequence()
    index_0=[i+1 for i,x in enumerate(c) if x==0]
    #print(c,index_0,len(c)) #show the doping sequence
    print(index_0)
    if c not in str_list:
        str_list.append(c)
    else:
        dupli_cout+=1
        if dupli_cout>40:
            print("maybe it has get the maxima str number!!")
            break
            
def factorial(n):
    s = 1
    for index in range(n, 0, -1):
        s *= index
    return s

# 定义排列函数
def A(n, k):
    return factorial(n) / factorial(n - k)

# 定义组合函数
def C(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

if doping_num<=len(srs_st):
    print(f"Total {doped_ele} number is :",len(srs))
    print(f"Unique {doped_ele} number is :{srs_st}")
    print(f"Unique {doped_ele} number is :",len(srs_st))
    print(f"Doping number on {doped_ele} is {doping_num}")
    print("Different structure number generated:",len(str_list),"\n")
    print(f"C {len(srs_st)} {doping_num} is {int(C(len(srs_st), doping_num))}")
else:
    print(f"Total {doped_ele} number is :",len(srs))
    print(f"Unique {doped_ele} number is :{srs_st}")
    print(f"Unique {doped_ele} number is :",len(srs_st))
    #print(f"Total {doped_ele} number is :{srs}")
    
    print(f"Doping number on {doped_ele} is {doping_num}")
    print("Different structure number generated:",len(str_list),"\n")
    print(f"C {len(srs)} {doping_num} is {int(C(len(srs), doping_num))}")

    
def create_str(file,filename):
    dir_f=os.path.join(os.path.expanduser("~"), file)
    if not os.path.exists("%sDoping_result"%dir_f.split(filename)[0]):
        os.mkdir("%sDoping_result"%dir_f.split(filename)[0])
    f1_original=open(dir_f).readlines()
    f1=[]#delete the empty rows
    #print("*****")
    for i in f1_original:
        if (len([x for x in i.split() if len(x)>0])==0):
            print("empty rows")
        else:
            f1.append(i)
    
    str_ele=[x for x in f1[5].split("\n")[0].split() if len(x)>0]
    str_num=[int(x) for x in f1[6].split("\n")[0].split() if len(x)>0]
    atom_tot=sum(str_num)
    #print(atom_tot)
    str_coord=[]
    for i in range(atom_tot):
        str_coord.append([float(x) for x in f1[i+8].split() if len(x)>3])
    #print(str_coord[0])
    doped_ele_range_l=0
    doped_ele_posi=str_ele.index(doped_ele)
    if doped_ele_posi!=0:
        for i in range(doped_ele_posi):
            doped_ele_range_l+=str_num[i]
            
    doped_ele_range_r=doped_ele_range_l+str_num[str_ele.index(doped_ele)]-1
    #print(doped_ele_range_l,doped_ele_range_r)
    fixed_coord=[]
    for i in range(len(str_coord)):
        if i not in range(doped_ele_range_l,doped_ele_range_r+1):
            fixed_coord.append(str_coord[i])
    #print(len(fixed_coord))
    fixed_ele=str_ele.copy()
    fixed_ele.remove(doped_ele)
    
    
    #create structure
    ele_tot=[]
    for i in str_list:
        f_new=f1[0:5].copy()
        #print(f_new)
    
        orig_seq_pos=[]
        doping_seq_pos=[]
        new_coord=[]
        fixed_coord_in=fixed_coord.copy()
        new_coord.extend(fixed_coord_in)
        #print(len(new_coord),len(new_coord[0]))
        #print(new_coord)
        #修改元素符号序列为dope和original元素
        for j in range(len(i)):
            if i[j]==1:
                orig_seq_pos.append(j)
            elif i[j]==0:
                if show_vac==1:
                    doping_seq_pos.append(j)
        #repositioning
        #print(orig_seq_pos)
        #print(doping_seq_pos)
        #print(len(orig_seq_pos))
        #print(len(doping_seq_pos))
        
        for k in orig_seq_pos:
            new_coord.append(str_coord[k+doped_ele_range_l])
        if show_vac==1:
            for k in doping_seq_pos:
                new_coord.append(str_coord[k+doped_ele_range_l])
            
        f_new.append(" ")
        f_new.append(" ")
        for j in range(len(fixed_ele)):
            f_new[5]+="   %s"%fixed_ele[j]
            f_new[6]+="   %d"%str_num[str_ele.index(fixed_ele[j])]
            
        if show_vac==0:
            f_new[5]+="   %s\n"%doped_ele
            #f_new[5]+="   %s\n"%doping_ele
            f_new[6]+="   %d\n"%(len(orig_seq_pos))
            #f_new[6]+="   %d\n"%(len(doping_seq_pos))   
        elif show_vac==1:
            f_new[5]+="   %s"%doped_ele
            f_new[5]+="   %s\n"%doping_ele
            f_new[6]+="   %d"%(len(orig_seq_pos))
            f_new[6]+="   %d\n"%(len(doping_seq_pos))
            
        f_new.append(f1[7])
        for j in range(len(new_coord)):
            f_new.append("   %.9f   %.9f   %.9f\n"%(new_coord[j][0],new_coord[j][1],new_coord[j][2]))
        f_out=open(f"%sDoping_result\\%s_drop_{doped_ele}_num_{i.index(0)+1}.vasp"%(dir_f.split(filename)[0],filename.split(".vasp")[0]),"w")
        for j in f_new:
            f_out.writelines(j)
        f_out.close()
    
    print("structures generated in %s"%dir_f.split(filename)[0])

create_str(file_dop,filename_dop)
create_str(file_orig,filename_orig)