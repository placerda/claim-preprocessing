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
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT")

## AOAI CONFIGURATION ##

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2023-03-15-preview" 
openai.api_key = AZURE_OPENAI_KEY

## AOAI MODELS AND ITS LIMITS ##

deployment_model = {
    "davinci": "text-davinci-003"
}
# tokens per prompt per model
prompt_limit = {
    "text-davinci-003": 4097
}
# requests per minute per model
requests_per_minute = { 
    "text-davinci-003": 120
}
# tokens per minute per model
tokens_per_minute = { 
    "text-davinci-003": 120
}

def complete(prompt_filename, variables, deployment="davinci", max_tokens=500, temperature=0.0, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1, stop=None):
    result = ""

    # load prompt_filename content into prompt
    with open(prompt_filename, 'r') as file:
        prompt = file.read()
        
    # replace variables
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    # check prompt length
    max_length = prompt_limit[deployment_model[deployment]] - max_tokens
    encoder = tiktoken.encoding_for_model(deployment_model[deployment])
    num_tokens = len(encoder.encode(prompt))
    if num_tokens > max_length:
        print(f"[INFO] prompt too long ({num_tokens}) for {deployment_model[deployment]} reducing to {max_length} - prompt: {prompt}")
        while(num_tokens > max_length):
            prompt = prompt[prompt.find("\n")+1:] #TODO: improve this
            num_tokens = len(encoder.encode(prompt))

    # do the completion
    try:
        prompt = openai.Completion.create(engine=deployment,prompt=prompt,temperature=temperature,max_tokens=max_tokens,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,best_of=best_of,stop=stop)
        result = prompt.choices[0].text
    except openai.error.RateLimitError  as e:
        count = 1
        while count < 11:
            try:
                sleep_time = 60
                print(f"[ERROR] reached aoai completion rate limit retrying for the {count} time waiting {sleep_time} sec - prompt: {prompt}")
                time.sleep(sleep_time)
                prompt = openai.Completion.create(engine=deployment,prompt=prompt,temperature=temperature,max_tokens=max_tokens,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,best_of=best_of,stop=stop)
                result = prompt.choices[0].text
                break
            except openai.error.RateLimitError  as e:
                count += 1
            except Exception as e:
                result = "error"
                print(f"[ERROR] aoai completion error: {e} - prompt: {prompt}")                
    except Exception as e:
        result = "error"
        print(f"[ERROR] aoai completion error: {e} - prompt: {prompt}")

    return result
