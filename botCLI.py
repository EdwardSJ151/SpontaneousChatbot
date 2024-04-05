import lmql
import LMQLsrc
import time


# Load in models
path = "/home/pressprexx/Code/Models/"
modelPath = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
lightModelPath = ""

lightModelFile = path + lightModelPath
modelFile = path + modelPath

model = lmql.model(f"local:llama.ccp:{modelFile}",
    cuda = True,
    n_ctx = 2048,
    verbose = False
)

lightModel = lmql.model(f"local:llama.cpp:{lightModelFile}", 
    tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    cuda = True,
    n_ctx = 2048,
    verbose = False
    # verbose=True
)


# Create variables
messages = {}
userName = ""
lastMessages = {}
lastMessage = {}
lastUserMessage = ""
lastBotMessage = ""
shortSummary = ""


# Helpful functions
def testPrompt(promptName, message):
    start = time.time()
    output = promptName(message)  # Assuming lmql handles this synchronously
    result = time.time() - start
    print(f'Time of {promptName.__name__}: {result:.2f}')
    print(output)


# Chatbot
def main():

    testPrompt(LMQLsrc.isOffensive, "Fuck off")
    print("\n")
    testPrompt(LMQLsrc.isQuestion, "You are great!")
    print("\n")
    testPrompt(LMQLsrc.isAskedToAbort, "I don't want to talk anymore")
    print("\n")

if __name__ == "__main__":
    main()