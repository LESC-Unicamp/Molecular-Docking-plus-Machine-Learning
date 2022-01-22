# #############################################################################
#  Code for the manuscript:
#
# Molecular docking coupled with machine learning to screen inhibitors 
# of SARS-CoV-2: A comprehensive analysis of the structure of the 
# potential ligands 
#
# Lilian Caroline Kramer Biasi, Opalina Vetrichelvan, 
# Luís Fernando Mercier Franco
#            
# This code reads the features of the molecules for the 
# machine learning using RDKit  
#                                                                              
# Version number: 1.0.0                                                        
# #############################################################################
#                      University of Campinas (Unicamp)              
#                      School of Chemical Engineering               
#                                                                              
#                 --------------------------------------           
#                  Developer:  Lilian Caroline Kramer Biasi
#                              Opalina Vetrichelvan
#                  Supervisor: Luís Fernando Mercier Franco         
#                 --------------------------------------     
#                  Created on February 3rd, 2021      
#                  Disponible online on January 19th, 2022     
#               
# #############################################################################
# Disclaimer note: 
# Authors assume no responsibility or liability for the use of this code.      
# #############################################################################
# Input file:
#
# - Excel file containing in each column: ID=df['ID']
#					  The name of the molecules (column named "Name");
#					  The SMILES of the molecules (column named "SMILES");
#					  The binding energy obtained with molecular docking 
#						(column named "Scored ligands (kcal/mol)").
#
# Output files:
# List of molecules and its features saved as 'fragcsvLipinski.csv' and
#  'fragLipinski.xlsx'
#
# #############################################################################


from subprocess import call
import numpy as np
import pandas as pd
import statistics
import time
import glob
import pybel

startTime = time.asctime()
startTimes = time.time()

##############################################################################                   
#                              Input information
##############################################################################

# Position of the grid box:
print("Enter the center of grid box (for X):")
x = float(input())
print("Enter the center of grid box (for Y):")
y = float(input())
print("Enter the center of grid box (for Z):")
z = float(input())

# Grid box size:
print("Enter the size of the grid box (for X):")
sx = float(input())
print("Enter the size of the grid box (for Y):")
sy = float(input())
print("Enter the size of the grid box (for Z):")
sz = float(input())

# Receptor:
print("Enter the name of the pdbqt file containing the receptor (without .pqbqt extension):")
receptor_name = str(input())+'.pdbqt'

# Exhaustiveness:
print("Enter the exhaustiveness value:")
exhaustiveness_value = float(input())


##############################################################################                   
#                             Ligands list
############################################################################## 
print("\n Running the simulations for components:\n")

#Creating a folder to save the results:
call("mkdir Results_"+receptor_name[:-6], shell=True)
   
#Reading the ligands name in the folder, and saving it to a txt:
lig=open(".\\Results_"+receptor_name[:-6]+"\\Ligand_file.txt","w")
lig_list=(glob.glob("*.pdbqt"))
lig_list.remove(receptor_name)
for l in lig_list:
    lig.write(l+"\n")
    
lig.close()

# To read the ligands name from a txt file:
Ligand_list=[]
with open(".\\Results_"+receptor_name[:-6]+"\\Ligand_file.txt","r") as f:
    for line in f:
        Ligand_list.append(line.strip())

# =============================================================================
# To read the ligands name from a cvv file:
# colnames = ["Ligand"]
# df = pd.read_csv("Ligand_list.csv", names=colnames, encoding="latin-1")
# Ligand_list = df["Ligand"].values.tolist()
# Ligand_list.remove("Ligand")
# =============================================================================


##############################################################################                   
#                             Running Vina
##############################################################################
#Creating the Result file:
result_file = open(".\\Results_"+receptor_name[:-6]+"\\Results.txt", "w") 
result_file.write("#################################################################\n")
result_file.write("#           Complete Results from Autodock Vina:                #\n")
result_file.write("#       Developed by Lilian and Opalina, 03-feb-2021            #\n")
result_file.write("#################################################################\n\n\n")
result_file.write("Simulation Information (Input data):\n")
result_file.write("Simulation started at: "+startTime +"\n")
result_file.write("Receptor: "+receptor_name+"\n")
result_file.write("Grid box size (x,y,z): ("+str(sx)+", "+str(sy)+", "+str(sz)+")\n")
result_file.write("Grid box position (x,y,z): ("+str(x)+", "+str(y)+", "+str(z)+")\n")
result_file.write("Exhaustiveness: "+str(exhaustiveness_value)+"\n\n\n")

    
# Running Vina:
Ligand_name=[]
Ligand_SMILES=[]
for l in Ligand_list:
    print(l)
    l_name=l[:-6]
    Ligand_name.append(l_name)
    call(".\\vina.exe --receptor " + receptor_name+" --ligand " + l+ 
         " --center_x "+str(x)+" --center_y "+str(y)+" --center_z "+str(z)+
         " --size_x "+str(sx)+" --size_y "+str(sy)+" --size_z "+str(sz)+
         " --exhaustiveness "+str(exhaustiveness_value)+" --out .\\Results_"+
         receptor_name[:-6]+"\\out_"+ l_name+".pdbqt --log .\\Results_"+
         receptor_name[:-6]+"\\log_"+ l_name+".txt", shell=True) 
    Ligand_SMILES.append(next(pybel.readfile("pdbqt", l)).write("smi").split("\t")[0])
    result_file.write("\n\n Component: "+ l_name +"\n")
    contents = open(".\\Results_"+receptor_name[:-6]+"\\log_"+ l_name+".txt").read()
    result_file.write(contents)
    
result_file.close() 


###############################################################################                   
#                    Saving the results in a summary
###############################################################################
#Creating a Summary of the Results:
in_file  = open(".\\Results_"+receptor_name[:-6]+"\\Results.txt", "r")
out_file = open(".\\Results_"+receptor_name[:-6]+"\\Summary.txt", "w") 
out_file.write("#################################################################\n")
out_file.write("#        Summary of the Results from Autodock Vina:             #\n")
out_file.write("#       Developed by Lilian and Opalina, 03-feb-2021            #\n")
out_file.write("#################################################################\n\n\n")
out_file.write("Simulation Information (Input data):\n")
out_file.write("Simulation started at: "+startTime +"\n")
out_file.write("Receptor: "+receptor_name+"\n")
out_file.write("Grid box size (x,y,z): ("+str(sx)+", "+str(sy)+", "+str(sz)+")\n")
out_file.write("Grid box position (x,y,z): ("+str(x)+", "+str(y)+", "+str(z)+")\n")
out_file.write("Exhaustiveness: "+str(exhaustiveness_value)+"\n\n\n")

out_file.write("        LIGAND                     BINDING ENERGY \n")
out_file.write("+--------------------+ +----------------------------------------+\n")
out_file.write("                            Minimum      Average        Median \n")
out_file.write("                          (kcal/mol)    (kcal/mol)    (kcal/mol) \n")
out_file.write("+--------------------+   +----------+  +----------+  +----------+\n")

               

val=[]
avg_val=[]
median_val=[]                                                                                                  
ligand_dict={} #dictionary where each key is the ligand, and each element is a tuple of (min,median,mean)
                                                                                           

lines = in_file.readlines()
for l in lines:
    flag=0
    #write the ligand name
    if l[1:10]==str("Component"):
        out_file.write(str(l[12:][:-1]).ljust(20))
        dict_component=str((l[12:][:-1]))
        
        
    if l[:4]==str("   1"):
        ligand_energy=[]
        val.append(float(l[12:18]))
        out_file.write(str(l[12:18]).rjust(15)) 
        flag=1
    #building list that just includes energies for specific ligand
            
    listy=["   1", "   2", "   3","   4", "   5", "   6", "   7","   8","   9"]
    if l[:4] in listy:
        ligand_energy.append(float(l[12:18]))
        
    #if i am done looking at this ligand, append its median and average
    if l[:4]==str("   9"):
        minimum=min(ligand_energy)
        mean=round(statistics.mean(ligand_energy),1)
        avg_val.append(mean)
        median=statistics.median(ligand_energy)
        median_val.append(median)
        ligand_dict[dict_component]=(minimum,median,mean)
        out_file.write(str(round(mean,1)).rjust(12)+ str(round(median,1)).rjust(13)+ "\n")
    
    #prints the word "Error" instead of the number if Vina was unable to read the pdbqt file    
    if l==str("Reading input ... \n"):
        out_file.write(str("Error").rjust(15)+str("Error").rjust(12)+ str("Error").rjust(13)+ "\n") 
        val.append(np.nan) #not a number (NaN)
        avg_val.append(np.nan)
        median_val.append(np.nan)
        
#prints the word "Error" instead of the number if Vina was unable to read the pdbqt file 
if lines[-1]==str("Reading input ... "):
    out_file.write(str("Error").rjust(15)+str("Error").rjust(12)+ str("Error").rjust(13)+ "\n") 
    val.append(np.nan)
    avg_val.append(np.nan)
    median_val.append(np.nan)


out_file.write("\n\n*****************************************************************\n")
#finding the molecule that has lowest binding energy, average, and median
key_list = list(ligand_dict.keys())
val_list = list(ligand_dict.values())
def get_key(my_dict, val,index):
    key_list=[]
    for key, value in my_dict.items():
         if val == value[index]:
             key_list.append(key)
 
    if len(key_list)==1:
        return key_list[0]
    elif len(key_list)>1:
        return key_list
    
out_file.write("*  Lowest binding energy: "+  str(min(val))+ " kcal/mol, Component: " + str(get_key(ligand_dict,min(val),0))+ "\n")
med=round(min(median_val),1)
out_file.write("*  Lowest median binding energy: " + str(med) + " kcal/mol, Component: "+ str(get_key(ligand_dict,min(median_val),1))+ "\n")   
mean_f=round(min(avg_val),1)  
out_file.write("*  Lowest average binding energy: " + str(mean_f) + " kcal/mol, Component: " +str(get_key(ligand_dict,min(avg_val),2)) + "\n")

out_file.write("*****************************************************************\n")

out_file.write("+---------------------------------------------------------------+\n")



#Execution time:
out_file.write("Simulation finished at: "+time.asctime()+"\n")     
executionTime = (time.time() - startTimes)/60
out_file.write("Execution time in min.: " + str(executionTime))

in_file.close()
out_file.close()


###############################################################################                   
#                        Saving the results to Excel
###############################################################################
writer = pd.ExcelWriter(".\\Results_"+receptor_name[:-6]+"\\Results.xlsx", 
                        engine="xlsxwriter")

inp=[["Simulation started at",startTime],["Receptor",receptor_name], 
     ["Grid box size (x,y,z)", "("+str(sx)+", "+str(sy)+", "+str(sz)+")"],
     ["Grid box position (x,y,z)", "("+str(x)+", "+str(y)+", "+str(z)+")"], 
     ["Exhaustiveness", exhaustiveness_value]]
df1 = pd.DataFrame (inp)
df1.columns=["Simulation Informations (Input data)", "Value"]
df1.to_excel(writer, sheet_name="Input", index=False)

resp=np.stack((Ligand_name, Ligand_SMILES, val, avg_val, median_val)).T
df2 = pd.DataFrame (resp)
df2.columns=["Ligand Name", "SMILES", "Lowest binding energy (kcal/mol)", "Lowest average binding energy (kcal/mol)", 
             "Lowest median binding energy (kcal/mol)"]
df2.to_excel(writer, sheet_name="Results")

writer.save()




print("\n Execution time in min.: " + str(executionTime)+"\n\n")
print("********************************************************************************************************")
print("** Thank you for using the script developed by Lilian Caroline Kramer Biasi and Opalina Vetrichelvan! **")
print("** In case of doubts and suggestions e-mail to: lilian.biasi@outlook.com                              **")
print("** Laboratory of Complex Systems Engineering - University of Campinas                                 **")
print("********************************************************************************************************")