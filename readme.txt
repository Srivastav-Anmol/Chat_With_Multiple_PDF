Required Applications:- VS Code, Anaconda/Miniconda(Create environment variables for the system which detects the Path of the variable), internet connection.
Follow the steps:-
a. Open the project using VsCode and run new terminal to create an environment by typing the following command:
>>conda create -p environmentVarName python==version(mainly greater than 3.9)
     >>conda activate environmentVarName/
     >>cls
b. Now to go https://makersuite.google.com/, login with your Google Id and create a API Key and copy it.
c. Create a file in project folder called as ".env" which will have API_KEY of Google_Gemini under the variable name:-- GOOGLE_API_KEY="your_api_key".
d. Run the terminal which will install all necessary libraries by writing the following command:-
     >>pip install -r requirements.txt
e. At last in the terminal type "streamlit run app.py" (without double quotes).
f. User can interact with multiple languages.

Note:- PDF file should be well text formatted otherwise it throw an error.

     

