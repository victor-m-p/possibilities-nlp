import os
import openai
from dotenv import load_dotenv # for api key management
load_dotenv()
import re

# function to clean answer from GPT-3
def extract_answers(sentence_str): # perhaps we can make this more general 
    sentence_lst = re.split("\d", sentence_str) # split by digits (i.e. response 1, 2, ..., n)
    sentence_lst = [x.strip() for x in sentence_lst] # remove trailing whitespace and newline. 
    sentence_lst = list(filter(None, sentence_lst)) # remove first (empty) string 
    sentence_lst = [x.replace(". ", "") for x in sentence_lst] # only remove starting ". " (not sure about general capitalization, and trailing full stop). 
    sentence_dct = {f"n{num}": ele for num, ele in enumerate(sentence_lst)} # dictionary to preserve order
    return sentence_dct
    
# function to query GPT-3 (takes the cleaning)
def gpt3(text): 
    
    # get api key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # run through GPT-3
    response = openai.Completion.create(
        model="text-davinci-002", 
        prompt=text, 
        temperature=0.7, 
        max_tokens=1000,
        frequency_penalty=0,
        presence_penalty=0)
    sentence_str = response.choices[0].text # extract full text
    
    # get sentence list 
    sentence_lst = extract_answers(sentence_str)
    
    return sentence_lst, sentence_str

### getting actions ### 
# Global variables
goal = "You set the goal of"
n_actions = 4

# test sentences
actions = [
    # large
    f"Imagine that you are in your thirties and you are unhappy with life. What are {n_actions} things you could do?",
    f"Imagine that you are in your thirties and you are unhappy with life. {goal} finding a romantic partner. What are {n_actions} things you could do?", # finding pleasure, finding meaning, ...
    # small
    f"Imagine that you are in your fifties and you have just been given a day off from work. What are {n_actions} things you could do?",
    f"Imagine that you are in your fifties and you have just been given a day off from work. {goal} of doing a road-trip. What are {n_actions} things you could do?", # doing something pleasurable, doing something meaningful, doing something normal...
]

# generate responses
action_lst = [gpt3(action) for action in actions]

action_lst[2] # sometimes point instead of numbers 

action_lst[0] # fails for this one... 
action_lst[1]

### getting outcomes ### 
# original prompt + Say you did (x). What are 3 things that could happen?

