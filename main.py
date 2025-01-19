import pandas as pd
from flask import Flask, render_template_string, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def load_data():
    file_path = 'Depression Student Dataset.csv'
    data = pd.read_csv(file_path)
    return data


# Train an ML model to predict depression
def train_model(data):
    # Select features and target
    X = data[['Heart Rate', 'Activity Level', 'Sleep Duration']]
    y = (data['Heart Rate'] > 90) & (data['Activity Level'] < 3000) & (data['Sleep Duration'] < 6)
    y = y.astype(int)  # Convert to binary labels (1 for Depressed, 0 for Not Depressed)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model, scaler


# Predict depression using the trained model
def predict_depression(model, scaler, heart_rate, activity_level, sleep_duration):
    input_features = scaler.transform([[heart_rate, activity_level, sleep_duration]])
    prediction = model.predict(input_features)
    return "Depressed" if prediction[0] == 1 else "Not Depressed"


# Flask app for rendering the results
app = Flask(__name__)

# Train the model and scaler
data = load_data()
model, scaler = train_model(data)


@app.route('/')
def home():
    return redirect(url_for('analysis'))


@app.route('/analysis')
def analysis():
    data = load_data()
    analysis = []
    for _, row in data.iterrows():
        status = predict_depression(model, scaler, row['Heart Rate'], row['Activity Level'], row['Sleep Duration'])
        analysis.append({
            "timestamp": row['Timestamp'],
            "heart_rate": row['Heart Rate'],
            "activity_level": row['Activity Level'],
            "sleep_duration": row['Sleep Duration'],
            "mood": row['Mood'],
            "depression_status": status
        })

    # Render the existing HTML template
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Depression Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            .entry {
                margin-bottom: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .depressed {
                background-color: #f8d7da;
            }
            .not-depressed {
                background-color: #d4edda;
            }
        </style>
    </head>
    <body>
        <h1>Depression Analysis Results</h1>
        <a href="/predict">Predict Depression</a> | <a href="/plots">View Plots</a>
        <hr>
        {% for result in analysis %}
        <div class="entry {{ 'depressed' if result['depression_status'] == 'Depressed' else 'not-depressed' }}">
            <p><strong>Timestamp:</strong> {{ result['timestamp'] }}</p>
            <p><strong>Heart Rate:</strong> {{ result['heart_rate'] }} bpm</p>
            <p><strong>Activity Level:</strong> {{ result['activity_level'] }} steps</p>
            <p><strong>Sleep Duration:</strong> {{ result['sleep_duration'] }} hours</p>
            <p><strong>Mood:</strong> {{ result['mood'] }}</p>
            <p><strong>Depression Status:</strong> {{ result['depression_status'] }}</p>
        </div>
        {% endfor %}
    </body>
    </html>
    '''
    return render_template_string(html_template, analysis=analysis)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        heart_rate = float(request.form['heart_rate'])
        activity_level = float(request.form['activity_level'])
        sleep_duration = float(request.form['sleep_duration'])

        prediction = predict_depression(model, scaler, heart_rate, activity_level, sleep_duration)

        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p><strong>Heart Rate:</strong> {{ heart_rate }} bpm</p>
            <p><strong>Activity Level:</strong> {{ activity_level }} steps</p>
            <p><strong>Sleep Duration:</strong> {{ sleep_duration }} hours</p>
            <p><strong>Depression Status:</strong> {{ prediction }}</p>
            <a href="/predict">Make Another Prediction</a> | <a href="/">Back to Home</a>
        </body>
        </html>
        ''', heart_rate=heart_rate, activity_level=activity_level, sleep_duration=sleep_duration, prediction=prediction)

    # Form HTML
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predict Depression</title>
    </head>
    <body>
        <h1>Predict Depression</h1>
        <form method="post">
            <label for="heart_rate">Heart Rate (bpm):</label><br>
            <input type="number" step="any" id="heart_rate" name="heart_rate" required><br><br>

            <label for="activity_level">Activity Level (steps):</label><br>
            <input type="number" step="any" id="activity_level" name="activity_level" required><br><br>

            <label for="sleep_duration">Sleep Duration (hours):</label><br>
            <input type="number" step="any" id="sleep_duration" name="sleep_duration" required><br><br>

            <button type="submit">Submit</button>
        </form>
        <a href="/">Back to Home</a>
    </body>
    </html>
    '''
    return render_template_string(html_template)


@app.route('/plots')
def plots():
    # Generate plots
    plt.figure(figsize=(10, 6))

    # Plot Heart Rate Distribution
    data['Heart Rate'].plot(kind='hist', bins=20, alpha=0.7, label='Heart Rate')
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.legend()
    plot_path = os.path.join('static', 'heart_rate_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Plot Activity Level Distribution
    plt.figure(figsize=(10, 6))
    data['Activity Level'].plot(kind='hist', bins=20, alpha=0.7, label='Activity Level')
    plt.title('Activity Level Distribution')
    plt.xlabel('Activity Level (steps)')
    plt.ylabel('Frequency')
    plt.legend()
    activity_plot_path = os.path.join('static', 'activity_plot.png')
    plt.savefig(activity_plot_path)
    plt.close()

    # Render plots in HTML
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Plots</title>
    </head>
    <body>
        <h1>Data Visualizations</h1>
        <a href="/analysis">Back to Analysis</a>
        <hr>
        <h2>Heart Rate Distribution</h2>
        <img src="/static/heart_rate_plot.png" alt="Heart Rate Distribution">
        <h2>Activity Level Distribution</h2>
        <img src="/static/activity_plot.png" alt="Activity Level Distribution">
    </body>
    </html>
    '''
    return render_template_string(html_template)


if __name__ == '__main__':
    # Ensure the static folder exists for saving plots
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
