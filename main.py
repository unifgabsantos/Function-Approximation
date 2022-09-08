import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
results = {
    "teste2":{"S1":[[],[]],"S2":[[],[]],"S3":[[],[]]},
    "teste3":{"S1":[[],[]],"S2":[[],[]],"S3":[[],[]]},
    "teste4":{"S1":[[],[]],"S2":[[],[]],"S3":[[],[]]},
    "teste5":{"S1":[[],[]],"S2":[[],[]],"S3":[[],[]]},
}

def Model(neurons:int,layers:int,interactions:int,fileName:str):
    file = np.load(fileName)
    x = file[0]
    y = np.ravel(file[1])
    regr = MLPRegressor(hidden_layer_sizes=([neurons]*layers),max_iter=interactions,activation='logistic',solver='adam',learning_rate = 'adaptive',n_iter_no_change=50)
    regr = regr.fit(x,y)
    y_est = regr.predict(x)
    return x,y,regr,y_est,fileName
def Results(data):
    x,y,regr,y_est,fileName = data[0],data[1],data[2],data[3],data[4]
    fileName = (fileName.split(".")[0])+".png"
    plt.figure(figsize=[14,7])
    plt.subplot(1,3,1)
    plt.plot(x,y)
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)
    return min(regr.loss_curve_),plt.gcf()


for i in range(2,6):
    for j in range(1,11):
        filename = f"teste{i}.npy"
        for k in range(3):
            if(k==0):
                loss,plot = Results(Model(50*j,2*j,200000,filename))
            elif(k==1):
                loss,plot = Results(Model(50*j,2*j,200000*j,filename))
            else:
                loss,plot = Results(Model(50*j,2*j,200000*j,filename))
            results[f"teste{i}"][f"S{k+1}"][0].append(loss)
            results[f"teste{i}"][f"S{k+1}"][1].append(plot)
file = open("Resultados.txt","w")        
for result in results.keys():
    for simulation in results[result].keys():
        temp  = results[result][simulation]
        i = temp[0].index(min(temp[0]))
        payload = f"{round(min(temp[0]),4)};{round(np.average(temp[0]),4)}\n"
        file.write(payload)
        temp[1][i].savefig(f"./Resultados/{result.capitalize()}/{simulation}.png")
file.close()