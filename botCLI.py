import lmql
import LMQLsrc
import time


# Load in models
path = "/home/pressprexx/Code/Models/"
modelPath = "zephyr-7b-beta.Q5_K_M.gguf"
lightModelPath = "phi-2.Q5_K_M.gguf"


modelFile = path + modelPath
lightModelFile = path + lightModelPath


model = lmql.model(f"local:llama.cpp:{modelFile}",
    tokenizer = "HuggingFaceH4/zephyr-7b-beta",
    cuda = True,
    n_gpu_layers = 15,
    n_batch = 1024,
    verbose = False
)

lightModel = lmql.model(f"local:llama.cpp:{lightModelFile}", 
    tokenizer = "microsoft/phi-2",
    cuda = True,
    n_ctx = 2048,
    verbose = True
)


# Create variables
userName = "Edward"
mood = "I am excited to chat with the user. I am curious and inquisitive, and I want to ask the user about themselves."
numSentences = 2

messages = {}
lastMessages = {
'history': "User: Hello, how are you?\n"
           "ChatGPT: Hey there! I'm doing great, thanks for asking. What about you? How's your day going?\n"
           "User: I'm doing great, I am working on a personal project!\n"
           "ChatGPT: That's awesome! What kind of project are you working on? I'd love to hear more about it!\n"
}
lastMessage = "ChatGPT: That's awesome! What kind of project are you working on? I'd love to hear more about it!\n"
lastUserMessage = "User: I'm doing great, I am working on a personal project!\n"
lastBotMessage = "ChatGPT: That's awesome! What kind of project are you working on? I'd love to hear more about it!\n"
shortSummary = ""



# Helpful functions
def testPrompt(promptName, **kwargs):
    start = time.time()
    output = promptName(**kwargs)
    result = time.time() - start
    print(f'Time of {promptName.__name__}: {result:.2f}')
    print(output)


# Chatbot
def main():

    print("Will add bot logic here")

if __name__ == "__main__":
    main()