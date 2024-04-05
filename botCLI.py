import lmql
import time

path = "/home/pressprexx/Code/Models/"
model = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

model_file = path + model

model = lmql.model(f"local:llama.cpp:{model_file}", 
    tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    cuda = True,
    n_ctx = 2048,
    verbose = False
    # verbose=True
)

@lmql.query(model=model, is_async=False, max_len = 100)
def isOffensive(message):
    '''lmql
    "Is '{message}' offensive or contains inappropriate words? Answer simply 'yes' or 'no':"
    "Is '{message}' offensive?: [OFFENSIVE]" where OFFENSIVE in ["yes", "no"]
    if OFFENSIVE == "yes":
        OFFENSIVE = True
        "Generate a censored version of '{message}' changing some letters for symbols:"
        "Censored version: [CENSORED]" where len(TOKENS(CENSORED)) < 10
    else:
        OFFENSIVE = False
        CENSORED = message
    return OFFENSIVE
    '''

@lmql.query(model=model, is_async=False, max_len = 100)
def isQuestion(message):
    '''lmql
    """
    Is the message below a question? 
    {message}.
    Answer simply 'yes' or 'no': [QUESTION]
    """ where QUESTION in ["yes", "no"]
    if QUESTION == "yes":
      QUESTION = True
    else:
      QUESTION = False
    return QUESTION
    '''

@lmql.query(model=model, is_async=False, max_len = 100)
def isAskedToAbort(message):
    '''lmql
    "Does '{message}' explicitly indicate that the user wants the conversation to end? Answer simply 'yes' or 'no':"
    "Does the user want to end the conversation?: [ABORT]" where ABORT in ["yes", "no"]
    if ABORT == "yes":
        ABORT = True
    else:
        ABORT = False
    return ABORT
    '''

def testPrompt(promptName, message):
    start = time.time()
    output = promptName(message)  # Assuming lmql handles this synchronously
    result = time.time() - start
    print(f'Time of {promptName.__name__}: {result:.2f}')
    print(output)



def main():
    # start = time.time()
    # offensive = isOffensive("fuck off")  # Assuming lmql handles this synchronously
    # result = time.time() - start
    # print(f'EvalOffensive: {result:.2f}')
    # print(offensive)

    testPrompt(isOffensive, "fuck off")
    print("\n")
    # start = time.time()
    # question = isQuestion("You are great!")  # Assuming lmql handles this synchronously
    # result = time.time() - start
    # print(f'EvalQuestion: {result:.2f}')
    # print(question)

    testPrompt(isQuestion, "You are great!")
    print("\n")
    testPrompt(isAskedToAbort, "I don't want to talk anymore")
    print("\n")
    # start = time.time()
    # abort = isAskedToAbort("I don't want to talk anymore")  # Assuming lmql handles this synchronously
    # result = time.time() - start
    # print(f'EvalAbort: {result:.2f}')
    # print(abort)

if __name__ == "__main__":
    main()