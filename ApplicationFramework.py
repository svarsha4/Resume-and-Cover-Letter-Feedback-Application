#Used the following sources for assistance:
#ChatGPT
#https://getbootstrap.com/docs/4.3/components/navbar/
#https://www.w3schools.com/howto/howto_html_file_upload_button.asp 
#https://stackoverflow.com/questions/18713528/how-to-check-if-the-the-user-has-uploaded-a-file-or-not-in-javascript
#https://youtu.be/PSEhNb69XpI?si=eQBlsQNe-oVfi8AW 
#https://youtu.be/mqhxxeeTbu0?si=pyJ6YWntR4oEOJHd
#https://youtu.be/xIgPMguqyws?si=l9FBRhHzR-zuLjOc
#https://youtu.be/4nzI4RKwb5I?si=3U5GkETVDgPZtU-x
#https://youtu.be/hqu5EYMLCUw?si=mij2gbcRhVelQSt3
#https://youtu.be/ggRohENBgek?si=n8yxsvYRYlyFgRe9
#https://www.jobscan.co/blog/powerful-verbs-that-will-make-your-resume-stand-out/
#https://youtu.be/8fVEMdHKmqM?si=tSTg2U7RpYhC_v_m
#https://www.indeed.com/career-advice/resumes-cover-letters/skills-for-a-data-scientist
#https://www.dataquest.io/blog/data-science-tools-for-beginners/
#https://www.techtarget.com/searchbusinessanalytics/feature/15-common-data-science-techniques-to-know-and-use
#https://www.springboard.com/blog/data-science/data-science-projects/
#https://www.glassdoor.com/blog/guide/software-engineer-skills/
#https://www.geeksforgeeks.org/software-engineering-projects/
#https://www.cloudzero.com/blog/software-engineering-tools/

#Received assistance from Adam Lear on getting tensorflow and transformers installed in my environment

#Note: The Base.html template ensures a consistent color and background theme for the web application.

#imports that enable web development
from flask import Flask, redirect, url_for, render_template, request
import PyPDF2
import os

#the pdf file of the resume or cover letter the user uploaded will be saved in the folder "uploads" that gets created
directory_name = "uploads"
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

#global variable that stores the contents of the resume or cover letter the user uploaded
resume_cov_contents = None

#imports that enable Natural Language Processing (NLP) techniques to be used for this project
#the NLP techniques used for this project are the Question Answering model, ROUGE to evaluate the accuracy of
#the question answering model for a given context, and the Context Free Grammar model
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import CFG
from nltk.parse.chart import ChartParser
from nltk import pos_tag
from nltk.grammar import CFG, Production
import re
nltk.download('averaged_perceptron_tagger')
from nltk.tree import Tree
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
rouge = evaluate.load("rouge")

#sets up the Question Answering model to be used for this project
#specifically, the t5-base model from the HuggingFace library is used
qa_model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

#defines the initial Context Free Grammars to be applied to the sentences of a cover letter to see if they make sense
context_free_grammars = CFG.fromstring("""
S -> NP VP
NP -> Det Nominal | Pronoun | Nominal N | Nominal PP | N
Nominal -> Nominal N | Nominal PP | N
VP -> V NP | X PP | V PP | V VP
X -> V NP
PP -> P NP | P VP
Det -> 'a' | 'an' | 'the' | 'this' | 'that' | 'my' | 'our'
Pronoun -> 'me' | 'I' | 'you' | 'it'
N -> 'data' | 'visualization' | 'machine' | 'learning' | 'statistics' | 'model' | 'communication' | 'teamwork' | 'science' | 'Kaggle' | 'Python' | 'Flask' | 'Java' | 'Tabeau' | 'HTML' | 'CSS' | 'Haskell' | 'Javascript' | 'design' | 'software' | 'technology' | 'technologies' | 'skills' | 'problem' | 'ability' | 'Visual Basic' | 'PowerBI' | 'API' | 'application' | 'applications' | 'web' | 'projects' | 'models' | 'resume' | 'cover' | 'letter' | 'creativity' | 'insight' | 'challenge' | 'automation' | 'scalability' | 'code' | 'program' | 'math' | 'experience' | 'process' | 'processes'
V -> 'is' | 'are' | 'were' | 'like' | 'need' | 'want' | 'do' | 'am' | 'interested' | 'learning' | 'designing' | 'creating' | 'implementing' | 'collaborating' | 'collaborated' | 'designed' | 'implemented' | 'automate' | 'initiate' | 'lead' | 'innovate' | 'innovated' | 'develop' | 'developed' | 'helping' | 'adjusting' | 'hope' | 'provide' | 'contribute' | 'foster' | 'investigate' | 'failed' | 'succeeded' | 'pursuing' | 'studying' | 'have' | 'work' | 'volunteer' | 'read' | 'wrote' | 'programmed' | 'coding' | 'modeling' | 'determined' | 'persevered' | 'administered' | 'assigned' | 'authorized' | 'coached' | 'coordinated' | 'directed' | 'empowered' | 'enabled'
P -> 'from' | 'to' | 'on' | 'near' | 'in' | 'about'                                      
""")


#This function then updates the Context Free
#Grammar rules for every unidentified word and part of speech tag that is provided
def update_grammar_rules(word, pos_tag):
    global context_free_grammars

    #if the part of speech tag is not identified for the given word, we will add it to the parse tree
    if pos_tag not in context_free_grammars._categories:
        context_free_grammars._categories.add(pos_tag)
    new_grammar_rule = Production(pos_tag, [word])
    productions = list(context_free_grammars.productions()) + [new_grammar_rule]
    context_free_grammars = CFG(start=context_free_grammars.start(), productions=productions)



#This function extracts all of the sentences from the cover letter the user uploaded
def extract_sentences(contents):
    sentences = sent_tokenize(contents)
    return sentences


#This function creates dynamic Context Free Grammars, which means that the context free grammars keep changing whenever a new
#sentence gets provided that has features not yet defined in the Context Free Grammars. From there, the provided sentence
#gets parsed using the newly updated Context Free Grammars in order to determine if it's legitimate.
#If a sentence is not legitimate, the user is provivded feedback to change that specific sentence in their cover letter
#so that the sentence becomes grammatically sound.
def dynamic_context_free_grammars(sentence):
    global context_free_grammars
    words = word_tokenize(sentence)  #tokenizes all the words in the given sentence
    parts_of_speech = pos_tag(words)  #finds the part of speech associated with each word

    #updates the parse tree accordingly when taking into account all the words and their
    #associated parts of speech tags in the sentence provided
    for word, tag in parts_of_speech:
        update_grammar_rules(word, tag)
    parser = ChartParser(context_free_grammars)
    
    #in order to determine if they are legitimate or not.
    #If the sentence is not legitimate, feedback is provided to the user to fix that sentence
    #to make it grammatically sound. Otherwise, no feedback for that particular sentence is provided
    if not list(parser.parse(words)):
        feedback = sentence
        return feedback
    return None


#This function creates a Question Answering model pertaining to four questions that are asked about the contents
#of the uploaded resume or cover letter.
#Specifically, the answers associated with the given contents of a resume or cover letter
#will be returned
def question_answering(context):
    global qa_model
    global fine_tuning_dataset
    questions = ["What kinds of skills are being described?", 
                 "What kinds of projects were worked on?", 
                 "What kinds of technologies were used?", 
                 "In what ways were the technologies described used?"]
    answers = []

    #for every question from the above questions list, the Question Answering model will generate an answer
    #corresponding to every question; these answers will all be stored in a list called answers, which will then
    #be returned
    for question in questions:
        input_text = "question: " + question + " context: " + context
        inputs = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True)
        answer = qa_model.generate(**inputs)
        answer_text = tokenizer.decode(answer[0], skip_special_tokens=True)
        answers.append(answer_text)
    return answers



#This function uses the ROUGE metric to evaluate the accuracy of the Question Answering model for the given contents
#of a resume or cover letter.
#When a user chooses a job from the dropdown menu on one of the pages in the web application, that value associated with
#the chosen job gets passed to this function.
#Depending on which job the user selected, the answers obtained from the contents of their resume/cover letter will be
#compared to the desired answers associated with applying for that job; this comparison is how the ROUGE metric yields
#an accuracy as to how the Question Answering model performed. From there, the overlap between the generated answers
#and expected answers will be used to provide feedback to the user as to what should still be kept on their resume/cover letter.
#Additionally, feedback will be provided to the user on how they can improve the content of their resume/cover letter
#by taking into account words that are present in the expected answers but not the generated answers.
def qa_to_specific_job(job, answers):

    #if the user would apply for a Data Science role, the expected_answers indicate things that look attractive on a resume
    #or cover letter pertaining to the Data Science role
    if job == "Data Scientist":
        feedbacks = []
        expected_answers = ["cloud computing statistics probability advanced mathematics machine learning data visualization database management data wrangling communication analytical thinking ",
                   "detecting anomalies personalizing suggestions evaluating machine learning models developing web applications",
                   "python tableau R visual basic PowerBI MySQL SAS Microsoft Excel Microsoft Access NoSQL AWS Azure",
                   "creating dashboards cleaning data writing queries creating statistical models creating machine learning models wrangling large amounts of data finding relationships between data using clustering normalization classification and regression models providing insight to facilitate smarter business decision making"]
    
    #if the user would apply for a Software Engineering role, the expected_answers indicate things that look attractive on a resume
    #or cover letter pertaining to the Software Engineering role
    elif job == "Software Engineer":
        feedbacks = []
        expected_answers = ["cloud computing application development user experience UX design systems analysis design automating processes designing scalable applications problem solving creativity communication data structures and algorithms understanding programming languages testing software attention to detail",
                   "designing web applications user experience UX design applications web scraping",
                   "Python Flask Java HTML CSS Haskell Javascript Figma Visual Basic C++ SmartDraw GitHub",
                   "Creating web applications developing software retrieving APIs web scraping enhancing user experience UX design creating user flow and wireframe diagrams prototyping making code automated and scalable"]
    
    #for every answer in the answers list (specifically the answers obtained from the contents of the resume/cover letter),
    #the ROUGE score will be obtained the accuracy based on the number of words that are similar between the expected_answer
    #and answer. 
    for i in range(len(answers)):

        rouge_scores = rouge.compute(predictions=[answers[i]], references=[expected_answers[i]])
        rouge1_score = rouge_scores['rouge1']

        #from there, we will obtain the words that overlap between the answers and expected_answers, and use that information
        #to let the user know the good things they have included on their resume or cover letter that should stay there
        answer_words = set(word_tokenize(answers[i].lower()))
        expected_answer_words = set(word_tokenize(expected_answers[i].lower()))
        overlapping_words = answer_words.intersection(expected_answer_words)
        unique_reference_words = expected_answer_words - answer_words


        #if there are any overlapping words, we make sure we let the user know they that should be kept on their resume
        #or cover letter; however, if the rouge accuracy is less than 70% feedback is provided as well; 
        #otherwise provide feedback to the user on how they can improve the content of their
        #resume/cover letter when taking into account the words that present in the expected_answer but not the
        #generated answer
        if overlapping_words:
            feedback = "Great job touching upon " + list(overlapping_words)[0] + ", ".join(list(overlapping_words)[:-1]) + ", and " + list(overlapping_words)[-1] + "!"
            feedbacks.append(feedback)
            if rouge1_score < 0.70 and i == 0:
                feedback = "Make sure you highlight " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1] + " as your skills"
                feedbacks.append(feedback)
            elif rouge1_score < 0.70 and i == 1:
                feedback = "Make sure you touch upon projects that involved " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1]
                feedbacks.append(feedback)
            elif rouge1_score < 0.70 and i == 2:
                feedback = "Make sure you are proficient in using the technologies " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1]
                feedbacks.append(feedback)
            elif rouge1_score < 0.70 and i == 3:
                feedback = "Make sure to provide descriptions, such as " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1] + " for how you used various technological tools"
                feedbacks.append(feedback)
        else:
            if i == 0:
                feedback = "Make sure you highlight " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1] + " as your skills"
                feedbacks.append(feedback)
            elif i == 1:
                feedback = "Make sure you touch upon projects that involved using " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1]
                feedbacks.append(feedback)
            elif i == 2:
                feedback = "Make sure you are proficient in using the technologies " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1]
                feedbacks.append(feedback)
            elif i == 3:
                feedback = "Make sure to provide descriptions, such as " + ", ".join(list(unique_reference_words)[:-1]) + ", and " + list(unique_reference_words)[-1] + " for how you used various technological tools"
                feedbacks.append(feedback)
    return feedbacks


#sets up the web application
app = Flask(__name__)


#This function represents the main page of the application, which is where the user uploads their resume or cover letter
#to be reviewed for feedback. The contents of the main page get generated by the home.html template.
#Additionally, once the user uploads their resume/cover letter, the contents of their uploaded document get portrayed at the
#bottom of the main page.
@app.route("/home", methods=["POST", "GET"])
def home():
    global resume_cov_contents
    uploaded_file = None
    if request.method == "POST":
        uploaded_file = request.files["file"]

        #once the resume/cover letter gets uploaded by the user, the contents of that document gets
        #retrieved and then displayed below the main page
        if uploaded_file:
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)
            pdf_contents = read_pdf(file_path)
            resume_cov_contents = pdf_contents
            return render_template("home.html", pdf_contents=pdf_contents, uploaded_file=uploaded_file)
    else:
        return render_template("home.html", uploaded_file=uploaded_file)


#This function specifically obtains the contents for the pdf file the user uploaded and gets called in the home()
#function.
def read_pdf(uploaded_file):
    pdf = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ''
    for page in pdf.pages:
        pdf_text += page.extract_text()
    pdf_text = str(pdf_text)
    return pdf_text


#This function directs the user to the page that allows them to verify whether they uploaded
#a resume or cover letter, as well as choose which job they 
#catered their resume/cover letter for.
#All of this information gets displayed on the selections.html template.
@app.route("/options/<file>", methods=["POST", "GET"])
def options(file):
    if request.method == "POST":
        feedbackName = request.form["option"]  #retrieves info as to whether the user uploaded a resume or cover letter
        jobName = request.form["job"]  #retrieves info as to which job their resume/cover letter is catered for
        return redirect(url_for("user", feedback=feedbackName, job=jobName))
    else:
        return render_template("selections.html")


#This function directs the user to the webpage that displays feedback corresponding to whether they
#uploaded a resume or cover letter, as well as the job they selected in the
#options() function. In order to display the approriate feedback, the functions question_answering and qa_specific_to_job
#get called. However, if the contents are specific to a cover letter, 
#there will also be feedback provided pertaining to proper grammar usage alongside
#feedback on the content usage. All of this information gets displayed in the feedback.html template.
@app.route("/user/<feedback>/<job>")
def user(feedback, job):
    global resume_cov_contents
    global qa_model
    qa_results = []
    grammar_results = []
    if feedback == "Cover Letter":
        sentences = extract_sentences(resume_cov_contents)
        for sentence in sentences:
            grammar_results.append(dynamic_context_free_grammars(sentence))
    answers = question_answering(resume_cov_contents)
    qa_results.append(qa_to_specific_job(job, answers))
    return render_template("feedback.html", feedback=feedback, qa_results=qa_results, grammar_results=grammar_results, resume_cov_contents=resume_cov_contents)


#runs the web application
if __name__ == "__main__":
    app.run(debug=True)