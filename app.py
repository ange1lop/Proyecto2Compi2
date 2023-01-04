import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split

st.title('Proyecto 2 201807299')
st.write('Escoga el algoritmo que desee')
prediccion = st.text_input('Dato a predecir', '0')
grado=st.number_input('Grado de la funcion (solo si es polinomial)',0)
#st.write('The current movie title is', title)
option = st.selectbox(
    'Algoritmos',
    ('Elija','Lineal', 'Polinomial', 'Arbol','Gauus','Redes neuronales'))

if (st.button('Proyectar')):
    if(option=='Lineal'):
        st.write('EStoy en lineal')
        data = pd.read_csv('Best_movies_netflix.csv')
        st.write(data)
        x = np.asanyarray(data['duration']).reshape(-1,1)
        y= data['number_of_votes']
        lineal = LinearRegression()
        lineal.fit(x,y)
        y_predict = lineal.predict(x)
        r2=r2_score(y,y_predict)
        st.write('R2: ',r2)
        coeficiente = lineal.coef_
        errorcuadrado = mean_squared_error(y,y_predict,squared=True)
        st.write('Error ',errorcuadrado)
        intercepto = lineal.intercept_
        funcion = "y="+str(round(coeficiente[0],3))+"x + ("+ str(round(intercepto,3))+')'
        st.write('Funcion: ',funcion)
        ynuevo = lineal.predict([[int(prediccion)]])
        st.write('Valor predicho',ynuevo)
        plt.title('Grafico de Votos en funcion de la duracion')
        plt.scatter(x,y)
        plt.plot(x,y_predict,color='r')
        plt.savefig("lineal.png")
        image = Image.open('lineal.png')
        st.image(image,caption='Grafico Regresion Lineal')
    elif(option=='Polinomial'):
        st.write('Estoy en Polinomial')
        datos = pd.read_csv('Best_movies_netflix.csv')
        st.write(datos)
        x=datos['duration'].values.reshape(-1,1)
        y=datos['number_of_votes'].values.reshape(-1,1)
        plt.scatter(x,y)
        poli=PolynomialFeatures(degree=int(grado))
        x_tranform = poli.fit_transform(x)
        model = LinearRegression().fit(x_tranform,y)
        y_new = model.predict(x_tranform)
        #Calculo de errores
        error = np.sqrt(mean_squared_error(y,y_new))
        coeficientes  = model.coef_
        funcion = ""
        contador = 0 
        for i in coeficientes[0]:
                funcion += str([i])+ "x^"+str(contador)
                contador += 1
        st.write('Funcion: y= ',funcion)
        r2=r2_score(y,y_new)
        st.write('R2: ',r2)
        st.write('Error: ',error)
        
        x_valor = np.array([int(prediccion)])
        x_v = x_valor.reshape(-1,1)
        x_vt = poli.fit_transform(x_v)
        yv= model.predict(x_vt)
        st.write('Prediccion',yv)


        x_nmin = -x.min()
        x_nmax = x.max()
        x_n=np.linspace(x_nmin,x_nmax,50)
        x_n = x_n.reshape(-1,1)
        x_ntransform = poli.fit_transform(x_n)
        y_new = model.predict(x_ntransform)
        plt.plot(x_n,y_new,color='r')
        plt.grid()
        plt.xlim(x.min(),x_nmax)
        plt.ylim(y.min(),y.max())
        
        titulo='Grado '+ str(grado)+'. Numero de Votos vs duracion'
        plt.title(titulo,fontsize=10)
        plt.savefig('polinomial.png')
        
        image = Image.open('polinomial.png')
        st.image(image,caption='Grafico de polinomial')
    elif(option=='Arbol'):
        st.write('Estoy en arbol')
        data3 = pd.read_csv('CancerBreast.csv')
        x= data3['radius_mean'].values
        x1= data3['texture_mean'].values
        x2= data3['perimeter_mean'].values
        x3= data3['area_mean'].values
        y= data3['diagnosis'].values
        le = LabelEncoder()
        x_encoded = le.fit_transform(x)
        y_encoded = le.fit_transform(y)
        features = list(zip(x_encoded))
        clf = DecisionTreeClassifier().fit(features,y_encoded)
        plot_tree(clf,filled=True)
        plt.savefig('arbol.png')
        texto = prediccion.split()
        enteros = []
        for t in texto:
            enteros.append(float(t))
        xk=np.array(enteros).reshape(-1,1)
        y_predict = clf.predict(xk)
        st.write('Prediccion: ',y_predict)
        image = Image.open('arbol.png')
        st.image(image,caption='Grafico del arbol')
    elif(option=='Gauus'):
        st.write('Estoy en Gauus')
        data4 = pd.read_csv('heart.csv')
        x = np.asanyarray(data4['age']).reshape(-1,1)
        y = np.asanyarray(data4['cp'])
        model = GaussianNB()
        model.fit(x,y)
        xpr = np.array(int(prediccion)).reshape(-1,1)
        ypredict = model.predict(xpr)
        st.write('Valor predicho',ypredict)
    elif(option=='Redes neuronales'):
        st.write('EStoy en Redes neuronales')
        data4 = pd.read_csv('Cellphonesdata.csv')
        x = np.array(data4['price']).reshape(-1,1)
        y = data4['performance']
        x_train,x_test,y_train, y_test = train_test_split(x,y)

        model = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3,3), random_state=1)
        model.fit(x_train,y_train)
        x_valor = np.array([int(prediccion)])
        x_v = x_valor.reshape(-1,1)
        yv= model.predict(x_v)
        st.write('Prediccion',yv)
    else:
        st.write('Ingrese un algoritmo')



