"""
---------------------------------------------------
This code is part of the AISTATS 2021 submission:
>>> Marienwald, Hannah, Fermanian, Jean-Baptiste & Blanchard, Gilles.
    "High-Dimensional Multi-Task Averaging and Application to Kernel 
    Mean Embedding." In International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021. <<<
---------------------------------------------------
"""

fig11 = (5,3)
fig11_wide = (8,4)

fsl = 10
fst = 15
liwi = 2.5


Colors = {'naive': 'k', 'STB': 'C0', 'MTA_const': 'C1', 'MTA_stb': 'C2', \
          'STB_weight': 'C3', 'RKMSE': 'grey', 'STB_theory': 'C4'}
Lstyle = {'Ridge': '-', 'Zeta': '-.', 'Gamma': ':', \
          'naive':'solid', 'STB':'solid', 'MTA_const':'solid', \
          'MTA_stb': 'solid', 'STB_weight': 'solid', 'RKMSE': 'solid', \
          'STB_theory': 'solid'}
Labels = {'naive':'Naive', 'STB':'STB-0', 'MTA_const':'MTA const', \
          'MTA_stb': 'MTA stb', 'STB_weight': 'STB weight', \
          'RKMSE': 'R-KMSE', 'STB_theory': 'STB theory'}
Marker = {'naive':'', 'STB':'x', 'MTA_const':'^', 'MTA_stb': 'v', \
          'STB_weight': '+', 'RKMSE': '', 'STB_theory': '*'}