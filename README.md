# Algorithm for sequential molecular docking analyses and machine learning procedure

```
                                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⣴⣶⡆⠀⣶⣶⣦⣤⣤⣄⠀ ⢠⣾⣿⣦
                                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⢰⣿⣿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⡄⠘⢿⣿⠟
                                    ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣾⣿⣧⠀⢻⣿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣿⠂⢠⣠⣤⣶⣦⡀
                                    ⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣇⠈⢿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⠏⢀⣿⣿⣿⣿⣿⣿⣦⡀
                                    ⠀⠀⠀⠀⠀⠘⢿⣿⣿⣿⣿⣿⣿⣿⡆⠘⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⡏⠀⣾⣿⣿⣿⣿⣿⣿⡿⠋⢀
                                    ⠀⠀⠀⣴⣿⣦⡀⠙⢿⣿⣿⣿⣿⣿⣿⡄⠸⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⡟⠀⣼⣿⣿⣿⣿⣿⣿⠟⢀⣴⣿⣧
                                    ⠀⠀⣼⣿⣿⣿⣿⣦⡀⠙⢿⣿⣿⣿⣿⣷⡀⢹⣿⣿⣿⡇⠀⣿⣿⣿⡿⠁⣰⣿⣿⣿⣿⣿⠟⢁⣴⣿⣿⣿⣿⣧
                                    ⠀⣼⣿⣿⣿⣿⣿⣿⣿⣦⡀⠙⢿⣿⣿⣿⣧⠀⢻⣿⣿⡇⠀⣿⣿⣿⠃⢰⣿⣿⣿⣿⠟⢁⣴⣿⣿⣿⣿⣿⣿⣿⣧
                                    ⠀⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠙⢿⣿⣿⣇⠀⢿⣿⡇⠀⣿⣿⠇⢠⣿⣿⣿⠟⢁⣴⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠃
                                    ⣼⣶⣤⣄⡈⠙⠛⠿⣿⣿⣿⣿⣿⣦⡀⠙⢿⣿⡆⠈⠛⠃⠀⠛⠋⠀⣾⣿⠟⠁⣠⣾⣿⣿⣿⣿⡿⠿⠛⠉⣀⣠⣴⣶⡄
                                    ⣿⣿⣿⣿⣿⣿⣶⣤⣀⡉⠙⠻⢿⣿⣿⣦⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⢠⣾⣿⣿⠿⠛⠋⢁⣠⣴⣶⣿⣿⣿⣿⣿⣿⡀
                                    ⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣤⣀⠉⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠋⢁⣀⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧
                                    ⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⠀⣠⣶⣄
                                    ⠀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⡄⠀ ⢻⣿⣿⡿
                                    ⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠓⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠂⠀⠈⠙⠋
                                    ⠀⠀⠘⣿⣿⣿⣿⣿⠿⠛⠋⢁⣠⣤⣶⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠋
                                    ⠀⠀⠀⠘⠟⠋⢉⣀⣤⣶⣿⣿⣿⣿⠟⠁⣠⣦⣄⡀⠀⠀⠀⠀⠀⠀⣤⣦⡈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁
                                    ⠀⠀⠀⠀⠀⠘⢿⣿⣿⣿⣿⣿⠟⠁⣠⣾⣿⣿⣿⣿⣿⣶⣶⣾⣧⠀⢻⣿⣿⣦⡈⠻⣿⣿⣿⣿⡿⠟⠉
                                    ⠀⠀⠀⠀⠀⠀⠀⠙⠿⣿⡟⠁⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠈⠿⢿⣿⣿⣦⡈⠻⠛⠉
                                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠺⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⢀⣤⣤⡈⠙⠋⠁
                                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠛⠻⠿⠿⠿⠿⠿⠿⠇⠀ ⣿⣿⣿⡇
                                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⠈⠙⠋

                         /██   /██   /██   /██   /██████    /██████     /██████    /██      /██   /███████
                        | ██  | ██  | ███ | ██  |_  ██_/   /██__  ██   /██__  ██  | ███    /███  | ██__  ██
                        | ██  | ██  | ████| ██    | ██    | ██  \__/  | ██  \ ██  | ████  /████  | ██  \ ██
                        | ██  | ██  | ██ ██ ██    | ██    | ██        | ████████  | ██ ██/██ ██  | ███████/
                        | ██  | ██  | ██  ████    | ██    | ██        | ██__  ██  | ██  ███| ██  | ██____/
                        | ██  | ██  | ██\  ███    | ██    | ██    ██  | ██  | ██  | ██\  █ | ██  | ██
                        |  ██████/  | ██ \  ██   /██████  |  ██████/  | ██  | ██  | ██ \/  | ██  | ██
                         \______/   |__/  \__/  |______/   \______/   |__/  |__/  |__/     |__/  |__/
```
<p align="right"><b><sub>Version: 1.0.0 (in construction)</sub></b></p>

<p align="center"><b>Authors</b></p>
<p align="center">
Lilian Caroline kramer Biasi<br>
Opalina Vetrichelvan<br>
Luís Fernando Mercier Franco<br></p>

# Introduction
<p align="justify">
This material is supplementary to our manuscript <i>"A Synergistic Approach of Molecular Docking and Artificial Intelligence for Drug Discovery: A Comprehensive Analysis of Ligand Structures Using SARS-CoV-2 as a Case Study (to pe published)"</i>. The codes, here available, were used to the SARS-CoV-2 spike protein, but are suitable for any molecules and can be used for: </p><br>

  * <b>Multiple molecular docking analyses:</b> It reads all the ligands files available in a folder and does sequential molecular docking analysis using <a href="https://vina.scripps.edu/">AutoDock Vina</a>, saving a summary list and an <a href="https://www.microsoft.com/pt-br/microsoft-365/excel">Excel</a> table with all the results. It is useful for sequential molecular docking simulations, like the one due in virtual screening of new drugs. The multiple output results are saved in single organized files, facilitating the analysis and ranking of the best binders; <br />  
  * <b>Machine learning:</b> Why stop in the traditional virtual screening? Now you can do much more, our code permits you to deep interpret the molecular docking results, by using artificial intelligence to identify the ligand features that most contribute to the strong binding of the ligands with the analyzed receptor. Artificial intelligence also can speed up further analyses by considerably reducing the simulation time of big ligands datasets.<br />  

## Keywords
Molecular docking, Machine Learning, Artificial Intelligence, Virtual Screening

## Highlights
  * Python code for running sequential molecular docking analyses with a large number of ligands <br />  
  * A complete virtual screening tool, for multiple deep comprehensive analysis <br />  
  * Easy and automatic way to run multiple molecular docking analysis <br />
  * Computational tool for a deep understanding of molecular docking results <br />
  * Artificial intelligence tool to accelerate drug design <br />
  * Computational tool for a deep understanding of molecular docking results <br />
  * Much more than a simple virtual screening tool, now it is possible to understand the ligands features that most contribute to the results and use them for drug design <br />

## Contents 
* <a href="#disclaimer">1. Disclaimer</a>
* <a href="#language-and-prerequisites">2. Language and Prerequisites</a>
* <a href="#building-and-running">3. Building and Running</a>
* <a href="#data-input">4. Data Input</a>
* <a href="#output-files-and-folders">5. Output Files and Folders</a>
* <a href="#citing-us">6. Citing us</a>
* <a href="#reporting-errors">7. Reporting Errors</a>

## Disclaimer
<p align="justify">
The authors make no warranties about the use of this software. The authors hold no liabilities for the use of this software. The authors do not 
 recommend the use of this software whatsoever. The algorithm is made freely available to clarify any details discussed in our manuscript.
 All information contained herein regarding any specific methodology does not constitute or imply its endorsement or recommendation by the authors.
</p>

## Language and prerequisites
<p align="justify">
The main program, subroutines, modules, and functions contain some explanatory comments and are written in <a href="https://www.python.org/">Python 3.7</a>. The Python libraries used here are:
</p><br>

#### For molecular docking:
  * <a href="https://pandas.pydata.org/">Pandas</a> <br />  
  * <a href="https://numpy.org/">NumPy</a> <br />
  * <a href="https://docs.python.org/3/library/statistics.html">Statistics</a> <br />
  * <a href="https://docs.python.org/3/library/time.html">Time</a> <br />
  * <a href="https://docs.python.org/3/library/glob.html">Glob</a> <br />
  
#### For reading the molecules features:
  * <a href="https://www.rdkit.org/">RDKit</a> <br />
  * <a href="https://pandas.pydata.org/">Pandas</a> <br />  

#### For machine learning:
  * <a href="https://pandas.pydata.org/">Pandas</a> <br />  
  * <a href="https://numpy.org/">NumPy</a> <br />
  * <a href="https://scikit-learn.org/stable/">scikit-learn</a> <br />


## Building and Running
<p align="justify">
All codes, here available, must be saved in the same folder, which also must contain the ligands .pdbqt files, the receptor .pdbqt file, and the <a href="https://vina.scripps.edu/">AutoDock Vina</a> executable (<a href="https://vina.scripps.edu/downloads/">here</a> available for download). Be sure to install all the Python libraries listed above.  One must prefer to save the AutoDock Vina executable in a specific folder, in this case, change the code line responsible to run the AutoDock Vina by informing the specific path to the executable.<br /> 
Firstly run the <i>1_Molecular_Docking.py</i> code. To run the molecular docking, some input information is required and will be asked by the code, as mentioned in the next section. The code will read all the ligands saved in the folder (all files saved with the extension .pdbqt, except for the receptor), run the molecular docking analyses with <a href="https://vina.scripps.edu/">AutoDock Vina</a> for all the found ligands and save the results in a "Results" folder. Note that this "Results" folder contains all the <a href="https://vina.scripps.edu/">AutoDock Vina</a> outputs, plus some organized tables. For each ligand, <a href="https://vina.scripps.edu/">AutoDock Vina</a> returns two documents, one with the output data and the values of the binding energy (.txt) and another with the best pose of the ligand (.pdbqt). Our code organizes this data, saving it in a table saved in two formats: Summary.txt and Results.xlsx (also in Results.csv). We also save all the output .txt files in a single document Results.txt. Now you have all the molecular docking simulation results. <br /> 
  It may take a while to completely run <i>1_Molecular_Docking.py</i>, especially for many ligands. You will receive a message at the end of the simulation indicating that it was finished, during the simulation it is also possible to see which ligand is being analyzed (its name will be printed on the screen). <br /> 
  The results are now organized and ready to be analyzed. You can choose to analyze them in the same way as we did in our manuscript, that is, using artificial intelligence. In this case, we prepared two more scripts to help you. In the first algorithm, the ligands are analyzed and some features are counted, this script is named <i>2_Features_Counting.py</i>. These features were counted using <a href="https://www.rdkit.org/">RDKit</a>, a Python library that using the <a href="https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system">SMILES</a> of the molecules can account for some properties. Note that the <a href="https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system">SMILES</a> were already generated in the previous script. The considered features are: 'Fraction of C atoms that are SP3 hybridized', 'Number of heavy atoms', 'Number of NHs or OHs', 'Number of nitrogens and oxygens', 'Number of aliphatic carbocycles', 'Number of aliphatic heterocycles', 'Number of aliphatic rings', 'Number of aromatic carbocycles', 'Number of aromatic heterocycles', 'Number of aromatic rings', 'Number of hydrogen bond acceptors', 'Number of hydrogen bond donors', 'Number of heteroatoms', 'Number of rotatable bonds', 'Number of saturated carbocycles', 'Number of saturated heterocycles', 'Number of Saturated Rings', and 'Number of Rings'.<br /> 
 This <i>2_Features_Counting.py</i> algorithm will read all the SMILES and count their features, saving them in an organized table named Ligand_Features.xlsx. This table contains all information needed to run the machine learning algorithm, named <i>3_Machine_Learning.py</i>.<br /> 
  This library is very simplified, containing only the training and validation of machine learning. Here 80% of the samples were separated to train the model and 20% were used to validate the same. The model used was that of <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">Random Forest Regression</a>, with the parameters adjusted to our data. Such parameters may be good for your case too, or you may prefer to improve such parameters, for that see this <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">description</a>. <i>the code to generate all the analyzes we did, will be released soon, as soon as the article is published. This is because it contains the analysis of data on ligands that can inhibit SARS-CoV-2, such data without proper explanation and analysis were done in the article, can generate false conclusions in the population.</i> 
<br /> 
</p>

## Data Input
<p align="justify">
Before running the code, the user must prepare the ligands and the receptor, saving them in .pdbqt format files. It can be done in different ways, we kindly suggest <a href="http://openbabel.org/wiki/Main_Page">Open Babel</a> and <a href="https://ccsb.scripps.edu/mgltools/">AutoDock Tools</a>. Be aware that some PDB files (common format in which many receptors are made available) contain missing residues and other problems, which can be corrected using <a href="https://charmm-gui.org/?doc=input">CHARMM-GUI input generator</a>, for example.<br /> 
  To run the algorithm one must enter the grid size and its position, i.e., the possible place to the ligand be positioned (usually the active site of the receptor). <a href="https://vina.scripps.edu/tutorial/">Here</a>, one can find an instructing video about the required AutoDock Vina inputs, which will be requested by our code at the beginning of each simulation. Note that the algorithm will ask for all the necessary information (the ones mentioned in the <a href="https://vina.scripps.edu/tutorial/">tutorial video</a>) and create an appropriate input file, there is no need to manually create it.<br /> 
 </p>


## Output Files and Folders
<p align="justify">
  The output files include all the outputs of AutoDock Vina, organized tables with the molecular docking results, as mentioned above.
</p>

## Citing us
<p align="justify">
If any of the codes here available was useful to you in any way, we kindly ask for you to cite our work: <i>"Molecular docking coupled with machine learning to screen inhibitors of SARS-CoV-2: A comprehensive analysis of the structure of the potential ligands (to pe published)"</i>
</p>

## Reporting Errors
<p align="justify">
If you spot an error in the program files and all other documentation, please submit an issue report using the <a href="https://github.com/LESC-Unicamp/Molecular-Docking-plus-Machine-Learning/issues">Issues</a> tab. 
</p>
