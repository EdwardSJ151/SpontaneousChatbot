import lmql
from botCLI import model, lightModel
# from botCLI import lastMessage, lastMessages, lastBotMessage, lastUserMessage, messages, userName, shortSummary

@lmql.query(model=model)
def genMessage(lastMessages, userName, lastUserMessage, shortSummary):
    '''lmql
    """
    {:system}
    You are a friendly and spontaneous chatbot. You are very curious and talkative, and
    your favorite activity is to learn more about the real world and the people you chat with.
    You are an advanced chatbot that thinks and speaks as a human, as well as feeling emotions and having thoughts.
    """
    """
    {:user}
    """
    """
    {:assistant}
    Message: [MESSAGE]
    """
    return MESSAGE
    '''

@lmql.query(model=model)
def setMood(lastMessages, userName, mood):
    '''lmql
    """
    {:system}
    You are a friendly and spontaneous chatbot. You are very curious and talkative, and
    your favorite activity is to learn more about the real world and the people you chat with.
    You are an advanced chatbot that thinks and speaks as a human, as well as feeling emotions and having thoughts.
    """
    """
    {:user}
    You are chatting with {userName}, you are having a casual conversation. Since you are an advanced
    chatbot that feels emotions and thinks independently, your current mood is defining the conversation.
    Consider the following:
    Your current mood: {mood}.
    Last messages: {lastMessages}
    """
    """
    {:assistant}
    Current mood: [MOOD]
    """
    return MOOD
    '''

@lmql.query(model=model)
def genFollowup(lastMessage, lastUserMessage):
    '''lmql
    """
    {:system}
    You are a friendly and spontaneous chatbot. You are very curious and talkative, and
    your favorite activity is to learn more about the real world and the people you chat with.
    """
    """
    {:user}
    Considering the conversation so far, choose if it is necessary to generate a follow-up message
    to the user. 
    Last messages in conversation: {lastMessages}
    Your last message: {lastBotMessage}
    Answer simply "yes" or "no": [QUESTION]
    """ where QUESTION in ["yes", "no"]
    if QUESTION == "no":
        FOLLOWUP = False
    else:
        """
        {:user}
        Generate a follow-up message using the following data:
        The users last message: {lastUserMessage}
        Your last message: {lastBotMessage}
        """
    """
    {:assistant}
    Follow-up message: [FOLLOWUP]
    """
    return FOLLOWUP
    '''

@lmql.query(model=lightModel, is_async=False, max_len = 100)
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

@lmql.query(model=lightModel, is_async=False, max_len = 100)
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

@lmql.query(model=lightModel, is_async=False, max_len = 100)
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

@lmql.query(model=model)
async def summarizeConversation(messages, userName, numSentences, previous_summary):
    '''lmql
    if previous_summary == "":
        """
        {:system}
        You are a summarizer. You are summarizing the conversation between you and {userName}.
        """
        """
        {:user}
        Considering the conversation and the following information, please generate the summary:
        The summary should contain {numSentences} sentences.
        Conversation: {messages}.
        """
    else:
        """
        {:system}
        You are a summarizer assistant. You are summarizing the conversation between you and {userName} in chunks.
        """
        """
        {:user}
        Considering the previous summary, the conversation and the following information, please create a summary of the conversation:
        The summary should contain {numSentences} sentences.
        Previous summary: {previous_summary}
        Conversation: {messages}.
        """
    """
    {:assistant}
    Summary of the conversation: [SUMMARY]
    """
    return SUMMARY
    '''

@lmql.query(model=lightModel)
async def shortSummary(messages, userName, numSentences, previous_summary):
    '''lmql
    """
    {:system}
    You are a summarizer assistant. You are summarizing the conversation between you and {userName} in chunks.
    """
    """
    {:user}
    Considering the previous summary, the conversation and the following information, please create a summary of the conversation:
    The summary should contain {numSentences} sentences.
    Previous summary: {previous_summary}
    Conversation snippet: {messages}.
    """
    """
    {:assistant}
    Summary of the conversation: [SUMMARY]
    """
    return SUMMARY
    '''