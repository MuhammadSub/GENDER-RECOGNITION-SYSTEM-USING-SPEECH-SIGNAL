# GENDER-RECOGNITION-SYSTEM-USING-SPEECH-SIGNAL
GENDER RECOGNITION SYSTEM USING SPEECH SIGNAL:
  Voice recognition technology has come a long way in recent years, and it can be broadly categorized into two types: isolated and continuous. Isolated speech recognition requires a clear pause between each word, whereas continuous speech recognition can understand speech without such breaks. Another important aspect of voice recognition is whether it is speaker-dependent or speakerindependent. A speaker-dependent system is tailored to a specific individual's voice, while a speakerindependent system is able to understand a variety of voices. In addition to these basic distinctions, voice recognition technology has a range of applications. One of the key uses of speaker recognition technology is for verification and authentication. This is when the speaker's voice is used to confirm their identity, such as when unlocking a phone or accessing a secure website. The other main application of voice recognition technology is identification, which involves using the speaker's voice to identify an unknown person. One of the most exciting things about voice recognition technology is that it continues to evolve and improve. With advancements in machine learning and artificial intelligence, we can expect voice recognition to become even more accurate and versatile in the future.
  
## Download Dataset:

http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/

### Methodology:
    1. Feature Extraction and Creation of Dataset:
        This script is a Python script that aims to classify audio files as belonging to male or female speakers based on their speech signal. It uses several libraries, including tarfile, pandas, numpy, librosa, matplotlib, and seaborn. It does the following:
        
        1. Extracts files from a compressed tar archive and places them in a destination folder.
        2. Iterates through a directory of files, opens a specific file, and reads lines from it until it finds a line that starts with "Gender:", then checks the value of that line and moves the audio files in the corresponding folder based on the gender value.
        3. Defines a function called get_mfcc, which takes in the audio signal and its sample rate as inputs, resamples the audio signal to a specific sample rate, extracts the Mel-frequency cepstral coefficients (MFCCs) of the signal, and returns these coefficients as a feature.
        4. Defines a function called get_pitch, which takes in the sample rate and audio signal as inputs, calculates the pitch of the signal, and returns it as a feature.
        5. Defines a function called extract_feature, which takes in the path of an audio file and its gender as inputs, extracts the MFCCs and pitch of the signal, and stores the features and the file's gender in a dataframe.
        6. Extracts features from the audio files and stores them in a dataframe.
        7. It utilizes two datasets: "https://www.kaggle.com/datasets/primaryobjects/voicegender" and "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/".

    2. Model Building and Evaluation and Prediction on Real Dataset:
        This script is a Python script that uses machine learning algorithms to classify audio files as belonging to male or female speakers based on their speech signal features. It uses several libraries, including numpy, pandas, and various modules from scikit-learn library. It does the following:
        
        1. Reads a csv file containing the extracted features and labels of the audio files.
        2. Drops the unnecessary columns from the dataframe.
        3. Splits the dataset into features and labels.
        4. Converts the dataframe to a numpy array.
        5. Splits the dataset into training and test sets.
        6. Trains several models: Decision Tree, Random Forest, Logistic Regression, and K-Nearest Neighbors (KNN).
        7. Make predictions on the test data using the trained models.
        8. Prints the accuracy of the models on the test data.
        9. Reads a csv file containing the extracted features of real-case audio files.
        10. Prints the accuracy of the Decision Tree model on the real-case data.
        11. It appears that the script is using a dataset of audio files to train and test several machine learning models, then it is using one of these models to classify real-case audio files.
        
    The given code is a Python script that is used to create a gender recognition system using speech signals. The system uses various machine learning models such as Decision Tree, Random Forest, Logistic Regression, and K-Nearest Neighbors (KNN) to classify a person's gender based on their speech.
    
    3. Data Visualization:
    
        1. The first step is to import the necessary libraries such as Pandas, Numpy, Scikit-learn, and Plotly.
        2. Next, the data is imported from a CSV file using the Pandas library. The data contains various speech signal features such as mean frequency, standard deviation, etc., along with the corresponding gender labels (male or female).
        3. The data is then split into training and testing sets using the train_test_split function from the Scikitlearn library.
        4. The next step is to train the various machine learning models on the training data. The Decision Tree, Random Forest, Logistic Regression, and KNN models are trained on the data using the corresponding functions from the Scikit-learn library.
        5. After the models are trained, they are used to predict the gender of the individuals in the testing data. The accuracy of each model is then calculated using the accuracy_score function from the Scikit-learn library.
        6. The next step is to create a data frame with the testing accuracy of all the models. The data frame is reshaped to a columnar format and then converted to a CSV file.
        7. The same process is repeated for the predicting accuracy of all the models.
        8. In the final step, the data from the CSV files is read using the Pandas library, and various visualizations such as bar charts, line charts, and scatter plots are created using the Plotly library to compare the accuracy of the different models.
        
### Findings:

  The results of our model analysis show that the Random Forest model had the highest testing accuracy at 95.3%, followed by the Decision Tree model with 91.0% accuracy. These models performed significantly better than the Logistic Regression model which had a lower testing accuracy of 28.2%, and the KNN model which had 86.3% accuracy. This suggests that the Random Forest and Decision Tree models are more effective at classifying the gender of individuals based on their speech
  characteristics. When it comes to predicting accuracy, the Decision Tree model still performed the best with 85.9% accuracy, while the Random Forest model had a lower accuracy of 77.2%. The Logistic Regression and KNN models had even lower predicting accuracies of 73.7% and 63.2%, respectively. This indicates that the Decision Tree model may be better at generalizing to new data and making accurate predictions. Overall, the Random Forest and Decision Tree models performed the best in both testing and predicting accuracy. It is important to note that these results should be considered in the context of the specific use case and requirements of the application. Additionally, further tuning and optimization of the models may improve their performance. As a next step, it's worth exploring more complex models such as Neural Networks, Convolutional Neural Networks, or Recurrent Neural Networks
