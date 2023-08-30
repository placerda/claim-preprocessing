"""
Title: Azure OpenAI Utils
Author: Paulo Lacerda
Description: Utility functions to use Azure OpenAI API

"""
import openai
import tiktoken
import time
import os

## Global variables ##

AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")

## AOAI CONFIGURATION ##

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2023-03-15-preview" 
openai.api_key = AZURE_OPENAI_KEY


def complete(prompt_filename, text, deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT, max_tokens=500, temperature=0.0, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    result = ""

    # load prompt_filename content into prompt
    with open(prompt_filename, 'r') as file:
        prompt = file.read()
    try:
        completion = openai.ChatCompletion.create(
            engine=deployment,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=temperature,max_tokens=max_tokens,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,stop=stop
        )
        result = completion.choices[0].message['content']        
    except openai.error.RateLimitError  as e:
        count = 1
        while count < 11:
            try:
                sleep_time = 60
                print(f"[ERROR] reached aoai completion rate limit retrying for the {count} time waiting {sleep_time} sec - prompt: {prompt}")
                time.sleep(sleep_time)
                completion = openai.ChatCompletion.create(
                    engine=deployment,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=temperature,max_tokens=max_tokens,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,stop=stop
                )
                result = completion.choices[0].message['content']   
                break
            except openai.error.RateLimitError  as e:
                count += 1
            except Exception as e:
                result = "error"
                print(f"[ERROR] aoai completion error: {e}")                
    except Exception as e:
        result = "error"
        print(f"[ERROR] aoai completion error: {e}")

    return result
