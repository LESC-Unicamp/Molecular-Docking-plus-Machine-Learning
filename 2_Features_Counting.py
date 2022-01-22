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
#                  Supervisor: Luís Fernando Mercier Franco         
#                 --------------------------------------     
#                  Created on June 2nd, 2021      
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


from rdkit import Chem
import pandas as pd
from rdkit.Chem import Lipinski as lip

# Reading the input information
print("Enter the name of the pdbqt file containing the receptor (without .pqbqt extension):")
receptor_name = str(input())

df = pd.read_excel('.\\Results_'+receptor_name+'\Results.xlsx', sheet_name='Results')

smiles=df['SMILES']
name=df['Ligand Name']
score=df['Lowest binding energy (kcal/mol)']


# Fragmentation of the molecule and features counting:
# For more information, see: https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html

dic={}

for index, row in df.iterrows():
    print('Reading SMILES: '+ row['SMILES'])
    s= row['SMILES']

    m = Chem.MolFromSmiles(s)
    dic[s]={}
    dic[s]['Ligand Name']= row['Ligand Name']
    dic[s]['Lowest binding energy (kcal/mol)']= row['Lowest binding energy (kcal/mol)']

    try:
        dic[s]['Fraction of C atoms that are SP3 hybridized']=lip.FractionCSP3(m)
        dic[s]['Number of heavy atoms']=lip.HeavyAtomCount(m)
        dic[s]['Number of NHs or OHs']=lip.NHOHCount(m)
        dic[s]['Number of nitrogens and oxygens']=lip.NOCount(m)
        dic[s]['Number of aliphatic carbocycles']=lip.NumAliphaticCarbocycles(m)
        dic[s]['Number of aliphatic heterocycles']=lip.NumAliphaticHeterocycles(m)
        dic[s]['Number of aliphatic rings']=lip.NumAliphaticRings(m)
        dic[s]['Number of aromatic carbocycles']=lip.NumAromaticCarbocycles(m)
        dic[s]['Number of aromatic heterocycles']=lip.NumAromaticHeterocycles(m)
        dic[s]['Number of aromatic rings']=lip.NumAromaticRings(m)
        dic[s]['Number of hydrogen bond acceptors']=lip.NumHAcceptors(m)
        dic[s]['Number of hydrogen bond donors']=lip.NumHDonors(m)
        dic[s]['Number of heteroatoms']=lip.NumHeteroatoms(m)
        dic[s]['Number of rotatable bonds']=lip.NumRotatableBonds(m)
        dic[s]['Number of saturated carbocycles']=lip.NumSaturatedCarbocycles(m)
        dic[s]['Number of saturated heterocycles']=lip.NumSaturatedHeterocycles(m)
        dic[s]['Number of Saturated Rings']=lip.NumSaturatedRings(m)
        dic[s]['Number of Rings']=lip.RingCount(m)
    except:
        dic[s]="Error"
        
        
# Saving the output:


dfdic=pd.DataFrame(dic).T
dfdic.index.names = ['SMILES']
dfdic.to_csv('.\\Results_'+receptor_name+'\\fragcsvLipinski.csv')
dfdic.to_excel('.\\Results_'+receptor_name+'\\fragLipinski.xlsx', sheet_name="Results") 