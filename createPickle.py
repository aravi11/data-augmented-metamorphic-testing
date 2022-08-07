import networkx as nx 
import collections
import csv
import pickle

mr_type = 'additive'
labelList = []
gList = []
gTestList = []
gTrainList = []
labelTrainList =[]
labelTestList = []
with open('./Labels/additive/additive_dataLabel_org_m1_m3_m4.csv', 'r') as csvFile:
	reader = csv.reader(csvFile)
	counter = 0
	for row in reader:
		print(row)
		dotFile = row[0]	
		label = row[1]
		filepath = './DotFiles/all/'
		G = nx.drawing.nx_pydot.read_dot(str(filepath)+ str(dotFile))
		#edge = nx.readwrite.edgelist.write_edgelist(G,str(counter)+".edgelist", data = False)
		#for x in edge:
		if counter > 35:
			gTestList.insert(counter,G)
			labelTestList.insert(counter,label)
			
		else:
			gTrainList.insert(counter,G)
			labelTrainList.insert(counter,label)

		gList.insert(counter,G)
		counter = counter + 1 
		labelList.insert(counter,label)	
		
	#zipbObj = zip(gList, labelList)
	#dataFile = dict(zipbObj)
	#for (k,v) in dataFile.items():
		#print(len(k.nodes),v) 


nx.write_gpickle(gList, "./gPickles/additive/additive_org_m1_m3_m4.gpickle")

with open('./Labels/additive/final_labels/additive_org_m1_m3_m4.txt', 'w') as f:
    for item in labelList:
        f.write("%s\n" % item)
csvFile.close()

