def isOffensive(message):
    '''lmql
    "Is '{message}' offensive or contains inappropriate words? Answer simply 'yes' or 'no':"
    "Is '{message}' offensive?: [OFFENSIVE]" where OFFENSIVE in ["yes", "no"]
    if OFFENSIVE == "yes":
        OFFENSIVE = True
        "Generate a censored version of '{message}' changing some letters for symbols:"
        "Censored version: [CENSORED]"
    else:
        OFFENSIVE = False
        CENSORED = message
    return OFFENSIVE
    '''

def evalIsAQuestion(message):
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


def evalIsAskedToAbort(message):
    '''lmql
    "Does '{message}' explicitly indicates that the user wants the conversation to end? Answer simply 'yes' or 'no':"
    "The user wants to end the conversation?: [ABORT]" where ABORT in ["yes", "no"]
    ABORT = getMostSimilarString(ABORT, ["yes", "no"])
    if ABORT == "yes":
        ABORT = True
    else:
        ABORT = False
    return ABORT
    '''