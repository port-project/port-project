from django.shortcuts import render
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create your views here.
def homeView(request):
    return render(request,'pages/welcome_page.html',{})

def dwellTime(request):
    return render(request,'pages/dwell_time.html',{})

def trtPage(request):
    return render(request,'pages/trt.html',{})

def analyseResult(request):
    
    if request.method == 'POST':
        age = request.POST.get('age')
        sex = request.POST.get('sex')
        cpt = request.POST.get('cpt')
        trest_bps = request.POST.get('trest_bps')
        chol = request.POST.get('chol')
        fbs = request.POST.get('fbs')
        rest_ecg = request.POST.get('rest-ecg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        old_peak = request.POST.get('old_peak')
        slope = request.POST.get('slope')
        ca = request.POST.get('ca')
        th_def_type = request.POST.get('thal')

        
           


        heart_data = pd.read_csv('heart_disease_data.csv')
        X = heart_data.drop(columns='target',axis=1)
        print(X)
        y = heart_data['target']
        X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , stratify=y , random_state=2)
        dataset_details = [int(age) , int(sex) , int(cpt) , int(trest_bps) , int(chol) , int(fbs), int(rest_ecg) , int(thalach) , int(exang) , float(old_peak) , int(slope) , int(ca) , int(th_def_type) ]
        print(dataset_details)
        model = LogisticRegression()
        model.fit(X_train , y_train)

        input_data = dataset_details

        # change the input data to a numpy array

        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the np array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshaped)
        if(prediction[0] == 0):
                st = 'The person doesn\'t have heart disease'
                return render(request,'pages/results.html',{'has_disease':st})
        else:
                st = 'The person has a heart disease'
                return render(request,'pages/results.html',{'has_disease':st})

        
    
    return render(request,'pages/analyse.html',{})

def resultsPage(request):
    return render(request,'pages/results.html',{})