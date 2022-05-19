import os
import numpy as np
import pandas as pd

def get_files(path,rule):
    all = []
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            if f.endswith(rule):
                all.append(f)
    return all

if not os.path.exists("./output/"):
    os.system('mkdir ./output/')

os.system('python heatmap.py')

os.system('python chord.py')

os.system('python network.py')
cell_type = np.loadtxt('./cell_type.csv',delimiter=",", dtype=str)
para1="./output/chord/chord_all.pdf"
para2="./output/CCImatix.csv"

#para5=("brown" "green" "cornflowerblue" "blueviolet")
os.system("Rscript chord.R"+" "+para1+" "+para2)

for i in range(len(cell_type)):
	#print(str(cell_type[i]))
	para3=str("./output/heatmap/Plot/"+str(cell_type[i])+".pdf")
	para4=str("./output/heatmap/File/"+str(cell_type[i])+".csv")
	os.system("Rscript heatmap.R"+" "+para3+" "+para4)

	para5=str("./output/chord/Plot/"+str(cell_type[i])+".pdf")
	para6=str("./output/chord/File/"+str(cell_type[i])+".csv")
	os.system("Rscript chord.R"+" "+para5+" "+para6)

para7="./output/CCImatix.csv"
para8="./output/bubble.pdf"
os.system("Rscript bubble.R"+" "+para7+" "+para8)

net_file = get_files(path='./output/Network/File', rule=".csv")

for i in range(len(net_file)):
	para9 = "./output/Network/File/"+str(net_file[i])
	para10 = "./output/Network/Plot/"+str(net_file[i].split(".")[0])+".pdf"
	os.system("Rscript network.R"+" "+para9+" "+para10)




