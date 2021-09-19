from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import pickle
import os
import sys
import win32api


from definition import *

# from flask.signals import request_tearing_down

app = Flask(__name__)

ResumeRankingModel=pickle.load(open('ResumeRanking.pkl','rb'))

# model = pickle.load(open('rankingResume.pkl', 'rb'))


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def createReq():
    if request.method == 'POST':
        if request.form['trainModel'] == 'trainModel':
            Reqtext = extract_text('./RequirementFile/Job description.pdf')
            reqExtractedData = [x.strip() for x in Reqtext]
            cleanedReqData = cleaningData(reqExtractedData)
            reqVector = creatingVector(cleanedReqData)
            # reqVectorOnlyCol = reqVector[0:0]
            
            # Now we will iterate all uploaded resume and pass to model
            allResume = request.files.getlist("allResume[]")
            for resume in allResume:
                resume.save(os.path.join('./AllResume', resume.filename))
            mypath=r'./AllResume'
            allTrainingFiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            for resume in allTrainingFiles:
                allRestext = extract_text(resume)
                resumeExtractedData = [x.strip() for x in allRestext]
                cleanedResData = cleaningData(resumeExtractedData)
                test_vector = converToVec(cleanedResData)
                frames = [reqVector, test_vector]
                reqVector = pd.concat(frames)
            
            prediction=ResumeRankingModel.predict(reqVector)
            # return reqVector.to_dict('split')
            return str(prediction)




















if __name__ == '__main__':
    app.run(debug=True)
