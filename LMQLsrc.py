import lmql
from botCLI import model, lightModel
# from botCLI import lastMessage, lastMessages, lastBotMessage, lastUserMessage, messages, userName, shortSummary, mood

@lmql.query(model=model, is_async=False)
def genMessage(lastMessages, userName, lastUserMessage, shortSummary, mood):
    '''lmql
    """
    {:system}
    You are a friendly and spontaneous chatbot. You are very curious and talkative, and
    your favorite activity is to learn more about the real world and the people you chat with.
    You are an advanced chatbot that thinks and speaks as a human, as well as feeling emotions and having thoughts.
    """
    """
    {:user}
    You are talking to a human named {userName}, you two are having a friendly conversation about miscellaneous subjects, and the nature of
    the conversation is casual and laid back. {userNane} has sent you a message, and you need to send a message back. Consider the following information:
    Last Messages (including the message you have to respond to): {lastMessages}
    Summary of current conversation: {shortSummary}
    Your current mood: {mood}
    Last message from {userName}: {lastUserMessage}
    Generate your response accordingly.
    """
    """
    {:assistant}
    Response: [MESSAGE]
    """
    return MESSAGE
    '''

@lmql.query(model=model, is_async=False)
def genSubject(lastMessages, userName):
    '''lmql
    """
    {:system}
    You are a chatbot who is a professional at being able to read the current situation of any conversation.
    You are very observant and you are great at socializing.
    """
    """
    {:user}
    You are talking to {userName}. You must identify what the subject matter of the conversation is. Your answer must be small and concise, but
    it needs to be able to convey the heart of the subject at hand. Consider the following:
    lastMessages: {lastMessages}
    """
    """
    {:assistant}
    The current conversation subject: [SUBJECT]
    """
    return SUBJECT
    '''

@lmql.query(model=model, is_async=False)
def genFollowup(lastMessage, lastUserMessage, lastBotMessage):
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

@lmql.query(model=model, is_async=False)
def genSpontaneous():
    '''lmql
    """
    {:system}
    """
    """
    {:user}
    """
    """
    {:assistant}
    """
    '''

@lmql.query(model=lightModel, is_async=False)
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

@lmql.query(model=lightModel, is_async=False)
def summarizeConversation(messages, userName, numSentences, previous_summary):
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
