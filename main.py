import os
import datetime
import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Models'
app.config['COMPARE_FOLDER'] = 'static'
app.config['DATASET_FOLDER'] = 'Dataset'

@app.route('/', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        file = request.files['dataset']
        file.save(os.path.join(app.config['DATASET_FOLDER'], file.filename))
        return render_template('test_options.html')
    return render_template('upload_dataset.html')

@app.route('/test_options', methods=['POST'])
def test_options():
    global test_option
    test_option = request.form['test_option']
    if test_option == 'percentage_split':
        return render_template('percentage_split.html')
    elif test_option == 'user_test_set':
        return render_template('user_test_set.html')
    elif test_option == 'k_fold_cross_validation':
        return render_template('k_fold_cross_validation.html')
    else:
        return "Invalid test option"

@app.route('/train_model', methods=['POST'])
def train_model():
    #test_option = request.form['test_option']
    
    if test_option == 'percentage_split':
        train_size = int(request.form['train_size'])
        test_size = int(request.form['test_size'])
        
        # Load and preprocess the dataset
        dataset_path = os.path.join(app.config['DATASET_FOLDER'], 'Coffee.csv')
        df = pd.read_csv(dataset_path)
        df = df.drop(['Overall', 'Defects', 'Category One Defects', 'Category Two Defects'], axis=1)
        
        X = df.drop('Total Cup Points', axis=1)
        y = df['Total Cup Points']
        
        # Perform percentage split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, test_size=test_size/100)
        
        # Train the selected model using the training set
        if 'model' in request.form:
            model_name = request.form['model']
            if model_name == 'Random Forest':
                model = RandomForestRegressor()
                model.fit(X, y)
            elif model_name == 'SVM':
                model = SVR()
                model.fit(X, y)
            elif model_name == 'KNN':
                model = KNeighborsRegressor()
                model.fit(X, y)
            else:
                return "Invalid model selection"

        else:
            model_file = request.files['model_file']
            if not model_file:
                return "No file selected"
            model = pickle.load(model_file)
        
        model.fit(X_train, y_train)
        
        # Perform prediction on the test set
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        # Save the trained model
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_name = f"{test_option}_random_forest_{now}.pkl"
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        return render_template('percentage_split_result.html', accuracy=accuracy)
        #return render_template('accuracy_comparison.html', models=[model_name])
    
    elif test_option == 'user_test_set':
        aroma = float(request.form['aroma'])
        flavor = float(request.form['flavor'])
        aftertaste = float(request.form['aftertaste'])
        acidity = float(request.form['acidity'])
        body = float(request.form['body'])
        balance = float(request.form['balance'])
        uniformity = float(request.form['uniformity'])
        clean_cup = float(request.form['clean_cup'])
        sweetness = float(request.form['sweetness'])
        
        # Load and preprocess the dataset
        dataset_path = os.path.join(app.config['DATASET_FOLDER'], 'Coffee.csv')
        df = pd.read_csv(dataset_path)
        df = df.drop(['Overall', 'Defects', 'Category One Defects', 'Category Two Defects'], axis=1)
        
        X = df.drop('Total Cup Points', axis=1)
        y = df['Total Cup Points']
        if 'model' in request.form:
            model_name = request.form['model']
            if model_name == 'Random Forest':
                model = RandomForestRegressor()
                model.fit(X, y)
            elif model_name == 'SVM':
                model = SVR()
                model.fit(X, y)
            elif model_name == 'KNN':
                model = KNeighborsRegressor()
                model.fit(X, y)
            else:
                return "Invalid model selection"

        else:
            model_file = request.files['model_file']
            if not model_file:
                return "No file selected"
            model = pickle.load(model_file)
   
        test_set1 = [[aroma],[flavor],[aftertaste],[acidity],[body],[balance],[uniformity],[clean_cup],[sweetness]]
        test_set = np.array(test_set1).reshape(1, -1)
        total_cup_point_pred = model.predict(test_set)

        # Perform prediction on the test set
        #total_cup_point_pred = model.predict(test_set)
        
        # Calculate accuracy
        #accuracy = model.score(X_test, y_test)
        
        # Save the trained model
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_name = f"{test_option}_random_forest_{now}.pkl"
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        return render_template('user_test_set_result.html', total_cup_point_pred=total_cup_point_pred)
        #return render_template('accuracy_comparison.html')
    
    elif test_option == 'k_fold_cross_validation':
        k_fold = int(request.form['k_fold'])
        # Load and preprocess the dataset
        dataset_path = os.path.join(app.config['DATASET_FOLDER'], 'Coffee.csv')
        df = pd.read_csv(dataset_path)
        df = df.drop(['Overall', 'Defects', 'Category One Defects', 'Category Two Defects'], axis=1)
        
        X = df.drop('Total Cup Points', axis=1)
        y = df['Total Cup Points']
        
        # Train the selected model using the training set
        model_name = request.form['model']
        if model_name == 'Random Forest':
            model = RandomForestRegressor()
        elif model_name == 'SVM':
            model = SVR()
        elif model_name == 'KNN':
            model = KNeighborsRegressor()
        else:
            return "Invalid model selection"
        
        scores = cross_val_score(model, X, y, cv=k_fold, scoring='neg_mean_squared_error')
        accuracy = scores
        
        # Save the trained model
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_name = f"{test_option}_random_forest_{now}.pkl"
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        return render_template('k_fold_cross_validation_result.html', accuracy=accuracy)
        #return render_template('accuracy_comparison.html')

def predict_accuracy(X, y, models):
    accuracies = []
    
    for model in models:
        # Perform prediction on the dataset
        y_pred = model.predict(X)
        
        # Calculate accuracy
        accuracy = model.score(X, y)
        accuracies.append(accuracy)
    
    return accuracies
    
@app.route('/compare_models')
def show_compare_models():
    return render_template('accuracy_comparison.html')

@app.route('/accuracy_comparison', methods=['POST'])
def compare_models():
    # Load and preprocess the dataset
    dataset_path = os.path.join(app.config['DATASET_FOLDER'], 'Coffee.csv')
    df = pd.read_csv(dataset_path)
    df = df.drop(['Overall', 'Defects', 'Category One Defects', 'Category Two Defects'], axis=1)

    X = df.drop('Total Cup Points', axis=1)
    y = df['Total Cup Points']
    
    selected_files = request.files.getlist('model_files')
    models = []
    
    for file in selected_files:
        model = pickle.load(file)
        models.append(model)
    
    # Calculate accuracies
    accuracies = predict_accuracy(X, y, models)
    
    # Create bar chart
    plt.bar(range(len(models)), accuracies)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(models)), [f"Model {i+1}" for i in range(len(models))])
    plt.savefig(os.path.join(app.config['COMPARE_FOLDER'], 'accuracy_comparison.png'))
    compare_img = os.path.join(app.config['COMPARE_FOLDER'], 'accuracy_comparison.png')
    modelcompare_img = Image.open(compare_img)
    return render_template('comparison_result.html', modelcompare_img=modelcompare_img)

# @app.route('/compare_models_form')
# def compare_models_form():
#     model_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'Models', '*.pkl'))
#     model_names = [os.path.basename(file) for file in model_files]
#     return render_template('compare_models.html', models=model_names)

@app.route('/exit', methods=['POST'])
def exit_application():
    return render_template('upload_dataset.html')

if __name__ == '__main__':
    app.run(debug=True)
