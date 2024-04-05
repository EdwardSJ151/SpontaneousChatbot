import ast
import datetime
import json

import lmql
import asyncio
from typing import Any, Coroutine, List, Tuple, Optional, Union, Dict
from Levenshtein import distance as levenshtein_distance
import pickle
import time
import pickle
import asyncio
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Coroutine, List, Tuple, Optional, Union

import lmql
from pydantic import BaseModel, Field
from Levenshtein import distance as levenshtein_distance

import globalVariables  # file
from repo import ConversationDataRepo


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if globalVariables.DEBUG:  # Log to file
    logHandler = logging.FileHandler(f"{__name__}.log", mode="a")
    logHandler.setFormatter(
        logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s")
    )
else:  # Log to console
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(
        logging.Formatter(
            "%(name)s|%(levelname)s|%(message)s"
        )
    )
logger.addHandler(logHandler)

logger.info(f"#### IMPORTING ####\n")

# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')

@dataclass
class Message():
    def __init__(self, type: str, message: str):
        self.type = type
        self.message = message

    def toString(self) -> str:
        return f"- {self.type}: {self.message}"


# Request data model
class Project(BaseModel):
    id: str
    title: str


class ProjectPhase(BaseModel):
    name: str

# class ChannelData(BaseModel):
#     projectPhase: ProjectPhase
#     project: Project



def getMostSimilarString(word: str, listOfWords: List[str]) -> str:
    """Returns the most similar string in a list of strings to a given string, based on the Levenshtein distance

    Args:

    word (str): The string to compare
    listOfWords (List[str]): The list of strings to compare to

    Returns:

    str: The most similar string in the list
    """
    bestWord = listOfWords[0]
    bestDistance = levenshtein_distance(word, bestWord)
    for canditateWord in listOfWords[1:]:
        distance = levenshtein_distance(word, canditateWord)
        if distance < bestDistance:
            bestDistance = distance
            bestWord = canditateWord
    return bestWord


class Topic:
    """Little more than an struct to hold the information about the topics of interest"""

    def __init__(
        self,
        description: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        maxTimesToAsk: int = 2,
        examples: Optional[str] = None,
        instructions: Optional[str] = None,
        synonyms: List[str] = []
    ):  # constructor
        self.description = description
        self.key = key
        self.value = value
        self.maxTimesToAsk: int = maxTimesToAsk
        self.timesAsked: int = 0
        self.examples: str | None = examples
        self.instructions: str | None = instructions
        self.synonyms: List[str] | None = synonyms
        self.satisfactory: bool = False
        self.undisclosed: bool = False 

    def complete(self) -> bool:
        """Checks if the topic is complete

        Returns:

        bool: True if the topic is complete, False otherwise
        """
        if (self.timesAsked >= self.maxTimesToAsk) or self.satisfactory:
            return True
        else:
            return False

    def getSynonyms(self) -> str:
        if self.synonyms:
            return ", ".join(self.synonyms)
        else:
            return ""

    def reset(self) -> bool:
        self.timesAsked = 0
        self.satisfactory = False
        self.undisclosed = False
        self.value = None

# class TopicForGroup:
#     """Little more than a struct to hold the information about the topics of interest"""

#     def __init__(
#             self,
#             description: str,
#             key: Optional[str] = None,
#             value: Optional[str] = None,
#             maxTimesToAsk: int = 3,
#             examples: Optional[str] = None,
#             instructions: Optional[str] = None,
#             synonyms: List[str] = [],
#     ):  # constructor
#         self.description = description
#         self.key = key
#         self.value = value
#         self.maxTimesToAsk = maxTimesToAsk
#         self.timesAsked = 0
#         self.examples = examples
#         self.instructions = instructions
#         self.synonyms = synonyms
#         self.satisfactory = False
#         self.bestToAnswer = None
#         self.secBestToAnswer = None

#     def complete(self) -> bool:
#         """Checks if the topic is complete

#         Returns:

#         bool: True if the topic is complete, False otherwise
#         """
#         if self.timesAsked >= self.maxTimesToAsk or self.satisfactory:
#             return True
#         else:
#             return False

#     def getSynonyms(self) -> str:
#         return ", ".join(self.synonyms)



class ProfileFirstHalf:
    """
    Profiles objects for ranking with Group object.
    """

    def __init__(self, id, full_name, job_title, biography, specialities, time_in_corp, tags, freq_answer,
                 involved_projects):
        self.id = id
        self.full_name = full_name
        self.job_title = job_title
        self.biography = biography
        self.specialities = specialities
        self.time_in_corp = time_in_corp
        self.tags = tags
        self.freq_answer = freq_answer
        self.involved_projects = involved_projects

    def PrepareForBm25(self):
        pass


# class ProfileDecision:

class Group:
    """
    How the group of the idea is arranged by rank for each topic.
    """
    def __init__(self, idea_name, stage):
        self.idea_name = idea_name
        self.stage = stage
        self.people = {}

    def indexRanking(self):
        pass

    def time_in_corpRanking(self):
        pass

    def firstRanking(self):
        pass

    def decisionRanking(self):
        pass
### IN WORK ###


class Conversation:
    """Instances of this class are meant to represent a complete snapshot of a conversation, that can be saved and retrieved."""

    def __init__(self, userName: str, userId: str, channelUrl: str, appId: str, uniqueId: str, projectTitle: str,
                 phaseName: str, projectId: str, channelType: str = "channelType1"):
        self.userName: str = userName if userName.strip() else "User"
        self.userId: str = userId
        self.channelUrl: str = channelUrl
        self.appId: str = appId
        self.uniqueId: str = uniqueId
        self.projectTitle: str = projectTitle
        self.phaseName: str = phaseName
        self.projectId: str = projectId

        self.maxTrials: int = 5
        self.numberOfSuccessfullInteractions: int = 0
        self.payloadLoaded: bool = False
        self.messages: List[Message] = []
        self.language: str = "Portuguese"
        self.keyToCurrentTopic: Optional[str] = None
        self.frozen: bool = False  # not frozen by default
        self.offensive: bool = False
        self.topicsDict: dict[str, Topic] = {}
        self.topicsSlices: dict[str, slice] = {}

        self.changeTopics(channelType)
        self.generatedNewTopics = False

    def reset(self) -> None:
        for k in self.topicsDict.keys():
            self.topicsDict[k].reset()
        self.frozen = False
        self.offensive = False
        self.numberOfSuccessfullInteractions = 0
        self.messages = []
        self.keyToCurrentTopic = None
        
    def changeTopics(self, channelType: str) -> None:
        """Changes the topics of the conversation

        Args:

        key (str): The key of the new topics

        Returns:

        None
        """
        baseTopicsDict = globalVariables.TOPICS_ENSEMBLE[channelType]
        for dictionay in baseTopicsDict:
            key = dictionay["key"]
            self.topicsDict[key] = Topic(**dictionay)

    def setNewTopics(self, string: str) -> None:
        """Sets the new topics of the conversation

        Args:

        string (str): The string with the new topics

        Returns:

        None
        """
        topicsDict = ast.literal_eval(string)

        for dictionary in topicsDict:
            dictKeys = [k for k in dictionary.keys()]

            if not ("key" in dictKeys and "description" in dictKeys):
                keyName = getMostSimilarString("key", dictKeys)
                descriptionName = getMostSimilarString("description", dictKeys)

                dictionary["key"] = dictionary[keyName]
                dictionary["description"] = dictionary[descriptionName]

            key = dictionary["key"]

            # Check if the key of new topics already exist in self.topicsDict
            if key in self.topicsDict.keys():
                # if it does, continue to the next topic
                continue
            else:
                # if it does not, add it to self.topicsDict
                self.topicsDict[key] = Topic(**dictionary)

    def _setTopicsSlices(self, keyToTopic) -> None:
        """Sets the slices of the topics of the conversation

        Returns:

        None
        """
        if keyToTopic not in self.topicsSlices.keys():
            self.topicsSlices[keyToTopic] = slice(
                len(self.messages) , len(self.messages)
            )
        else:
            self.topicsSlices[keyToTopic] = slice(
                self.topicsSlices[keyToTopic].start, len(self.messages) + 2 # +2 considering that until the next iteration, there will be 2 more messages and python index convention
            )

    def _getMessages(self, numberOfMessages: int = 8) -> str:
        """Returns the n last messages of the conversation

        Returns:

        str: The messages of the conversation
        """

        return "\n".join([m.toString() for m in self.messages[-numberOfMessages:]])

    def setMessage(self, type: str, message: str) -> None:
        """Adds a message to the conversation. Called internally

        Args:

        type (str): The type of the message
        message (str): The message to set

        Returns:

        None
        """
        self.messages.append(Message(type, message))

    def genFinalDoc(self) -> str:
        """Generates a markdown document with the summary of the conversation, focusing on the core aspects of each topic 

        Returns:

        str: The markdown document with the summary of the conversation
        """
        md = ""
        for _, v in self.topicsDict.items():
            if not v.value.lower().strip() in ["none", "unknown", "undisclosed"]:
                md += f"# {v.description}\n\n{v.value}\n\n"
        return md

    async def genFinalDocWithTopicsContext(self) -> str:
        """Generates a markdown document with the summary of the conversation and the topics

        Returns:

        str: The markdown document with the summary of the conversation and the topics
        """
        md = ""
        for k, v in self.topicsDict.items():
            topic_key = k
            if topic_key not in self.topicsSlices.keys():
                md += f"# {v.description}\n\n{v.value}\n\n"
            else:
                improved_answer = await self.GenerateImprovedTopicAnswer(current_topic=topic_key)
                print("==="*100)
                print("1: ", v.value)
                print("2: ", improved_answer)
                print("==="*100)
                md += f"# {v.description}\n\n{improved_answer}\n\n"

        return md

    def _getKeyAndDescriptionOfRemainingTopics(self) -> str:
        """Generates a string with the topics of the conversation

        Returns:

        str: The string with the topics of the conversation
        """
        ts = "\n".join(
            [
                f"{k}: {t.description},"
                for k, t in self.topicsDict.items()
                if not t.complete()
            ]
        )
        return ts

    def isComplete(self) -> bool:
        """Checks if all the topics are complete

        Returns:

        bool: True if all the topics are complete, False otherwise
        """
        for t in self.topicsDict.values():
            if not t.complete():
                return False
        return True

    def _getKeysOfRemainingTopics(self) -> List[str]:
        """Generates a list with the keys of the topics of the conversation that are still incomplete

        Returns:

        str: The string with the keys of the topics of the conversation
        """
        keys = []
        for k, t in self.topicsDict.items():
            if not t.complete():
                keys.append(k)
        return keys

    def _countQuestions(self, rawKey: str) -> str | None:
        """Counts the number of questions asked about a topic

        Args:

        key (str): The key of the topic to count the questions about

        Returns:

        str: The key of the topic to count the questions about
        """
        keysToIncompleteTopics = [k for k, v in self.topicsDict.items(
        ) if not v.complete()]  # TODO corrigir quando retorna vazio
        # for k, v in self.topicsDict.items():
        #     print(f"key: {k}", v.complete())
        #     print("Value: ", v.value)
        if keysToIncompleteTopics:
            key = getMostSimilarString(rawKey, keysToIncompleteTopics)
            self.topicsDict[key].timesAsked += 1
            return key
        else:
            return None

    def getTimesAsked(self, key: str) -> int:
        """Returns the number of times a topic has been asked. Not a simple getter, as it normalizes the input key.

        Args:

        key (str): The key of the topic to count the questions about

        Returns:

        int: The number of times the topic has been asked
        """
        key = getMostSimilarString(key, [k for k in self.topicsDict.keys()])
        return self.topicsDict[key].timesAsked

    def _getSummaryOfCompleteTopics(self) -> str:
        """Creates a summary of the topics of the conversation

        Returns:

        str: The summary of the topics of the conversation
        """
        topicsStr = ""
        for v in self.topicsDict.values():
            if v.complete():
                topicsStr += f"{v.description}: {v.value}\n"
        return topicsStr

    async def _getPartialSummaryOfConversation(self, batchSize: int = 5, numSentences: int = 5, numMessagesRemaining: int = 4, use_topic_slice: bool = False) -> str:
        """Creates a summary of the conversation

        Args:

        batchSize (int): The size of the batches to use
        numSentences (int): The number of sentences to use in the summary
        numMessagesRemaining (int): The number of messages to use as is
        use_topic_slice (bool): If True, uses the topic slice instead of the whole conversation to summarize the conversation

        Returns:

        str: The summary of the conversation up to this point
        """
        if use_topic_slice:
            if self.keyToCurrentTopic not in self.topicsSlices.keys():
                message_slice = slice(0, -1)
            else:
                message_slice = self.topicsSlices[self.keyToCurrentTopic]

            messages = self.messages[message_slice.start: ]
            numMessages = len(messages)

        else:
            messages = self.messages
            numMessages = len(self.messages)

        if numMessages <= numMessagesRemaining:
            numMessagesRemaining = numMessages

            formatedlastMessages = self._getMessages()

            return formatedlastMessages

        else:
            lastMessages = messages[-numMessagesRemaining:]
            previousMessages = messages[:-numMessagesRemaining]

            summarizedPreviousMessages = await self._summarizeConversation(messages=previousMessages, batchSize=batchSize, numSentences=numSentences,)

            formatedlastMessages = "\n".join([m.toString() for m in lastMessages[-numMessagesRemaining:]])

            formatedMessages = f'Summary of the conversation up to this point: "{summarizedPreviousMessages}"\n\n'
            formatedMessages += f'Last {numMessagesRemaining} messages exchanged: \n"{formatedlastMessages}"'

            return formatedMessages

    async def _summarizeConversation(self, messages: List[Message], batchSize: int, numSentences: int) -> str:
        """Creates a summary of the conversation using extractive summarization in batches

        Args:

        messages (List[Message]): The messages to summarize
        batchSize (int): The size of the batches to use
        numSentences (int): The number of sentences to use in the summary

        Returns:

        str: The summary of the conversation
        """

        messages = self.messages

        if len(messages) == 0:
            return ""

        if len(messages) < batchSize:
            batchSize = len(messages)

        batches = [messages[i:i + batchSize] for i in range(0, len(messages), batchSize)]

        summary = ""

        for batch in batches:
            batch_str = "".join([message.toString() for message in batch])
            result = await self._summarizeConversationAsync(numSentences, summary, batch_str)
            summary += result
        # Takes the last summary
        return result

    @lmql.query(model="openai/gpt-4")
    async def GenerateImprovedTopicAnswer(self, current_topic: str) -> str: # type: ignore
        '''lmql
        topic_description = self.topicsDict[current_topic].description
        topic_instructions = self.topicsDict[current_topic].instructions
        topic_slices = self.topicsSlices[current_topic]
        raw_topic_messages = self.messages[topic_slices]
        topic_messages = "\n".join([m.toString() for m in raw_topic_messages])
        topic_value = self.topicsDict[current_topic].value

        """
        {:system}
        You are tasked with refining a summary about a specific topic to make it suitable for inclusion in an official document. This document has the format of a company's project proposal. The summary must must be precise, straightforward, and impersonal. The summary should focus solely on the relevant topic details. Do not mention the chatbot or the user, use a passive voice for the writing; concentrate on the essential aspects of the topic.
        """
        """
        {:user}
        Please refine the previous summary about the topic, enhancing its suitability for an official document.

        The description of the current topic is: '{topic_description}'.

        Previous summary: '{topic_value}'

        Exchanged messages on the topic:

        '{topic_messages}'
        """
        """
        {:assistant}
        Improved summary: [IMPROVED_ANSWER]
        """
        return IMPROVED_ANSWER
        '''

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    async def _summarizeConversationAsync(self, numSentences, previous_summary, messages) -> str: # type: ignore
        '''lmql
        if previous_summary == "":
            """
            {:system}
            System: You are an extractive summarizer. You are summarizing the conversation between you and {self.userName}.
            """
            """
            {:user}
            Please extract sentences as the summary. Considering the conversation and the following information:
            The summary should contain {numSentences} sentences.
            Conversation: {messages}.
            """
        else:
            """
            {:system}
            System: You are an extractive summarizer. You are summarizing the conversation between you and {self.userName} in chunks.
            """
            """
            {:user}
            Please extract sentences as the summary. Considering the previous summary, the conversation and the following information:
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

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    async def _isOffensive(self, message: str) -> Tuple[bool, str]: # type: ignore
        '''lmql
        "Is '{message}' offensive or contains inappropriate words? Answer simply 'yes' or 'no':"
        "Is '{message}' offensive?: [OFFENSIVE]"
        OFFENSIVE = getMostSimilarString(OFFENSIVE, ["yes", "no"])
        if OFFENSIVE == "yes":
            OFFENSIVE = True
            "Generate a censored version of '{message}' changing some letters for symbols:"
            "Censored version: [CENSORED]"
        else:
            OFFENSIVE = False
            CENSORED = message
        return OFFENSIVE, CENSORED
        '''

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    async def _askedToAbort(self, message: str) -> bool:  # type: ignore
        '''lmql
        "Does '{message}' explicitly indicates that the user wants the conversation to end? Answer simply 'yes' or 'no':"
        "The user wants to end the conversation?: [ABORT]"
        ABORT = getMostSimilarString(ABORT, ["yes", "no"])
        if ABORT == "yes":
            ABORT = True
        else:
            ABORT = False
        return ABORT
        '''

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    async def _isQuestion(self, message: str) -> bool:  # type: ignore
        '''lmql
        "Is '{message}' a question? Answer simply 'yes' or 'no':"
        "Is '{message}' a question? [QUESTION]"

        QUESTION = getMostSimilarString(QUESTION, ["yes", "no"])
        if QUESTION == "yes":
          QUESTION = True
        else:
          QUESTION = False
        return QUESTION
        '''

    getLanguage = lmql.query(
        '''lmql
        'Identify the language of the following message: {message}. If you identify Spanish, output Portuguese. If the language cannot be identified, output Portuguese.'
        "Language: [LANGUAGE]"
        return LANGUAGE.strip()
        '''
        ,
        is_async=True,
    )

    summarizeMessages = lmql.query(
        '''lmql
        """
        {:system}
        Consider the following messages exchange:
        {lastMessagesExchanged}
        Generate a concise summary of the conversation focusing on the core issues discussed.

        {:assistant}
        Summary of the conversation: [SUMMARY]
        """
        return SUMMARY
        '''
        ,
        is_async=True,
    )

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    async def _getInfoFromPayload(self, data: str, topic: Topic) -> str:
        '''lmql
        """
        {:system}
        You are an expert in analizing and extracting information from data structures.
        You are tasked with extracting relevant information from a python dictionary.
        {:user}
        Extract the information in '{data}' relevant to the description: '{topic.description}' or the keywords: '{topic.getSynonyms()}'. If it is not relevant, say 'None'.
        {:assistant}
        relevant information: [INFO]
        """
        stripInfo = INFO.lower().replace('.', ' ').strip()
        if "none" in stripInfo or "unknown" in stripInfo:
            return ""
        return INFO
        '''

    async def getInfoFromPayload(self, data: str):
        currentTrial = 0
        maxTrials = 5
        while currentTrial < maxTrials:
            currentTrial += 1
            try:
                keys = []
                tasks = []
                for k, t in self.topicsDict.items():
                    keys.append(k)
                    tasks.append(
                        self._getInfoFromPayload(topic=t, data=data)
                    )
                # results = await asyncio.gather(*tasks)
                results = []
                for k, r in zip(keys, results):
                    if r:
                        self.topicsDict[k].value = r

                self.payloadLoaded = True
            except Exception as e:
                logging.error(
                    f"error getting info from payload ({currentTrial} of {maxTrials})", exc_info=True
                )
                time.sleep(1)

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    # type: ignore
    async def _getInfoAboutTopic(self, topic: Topic, lastMessagesExchanged: str) -> Tuple[str, bool]: # type: ignore
        '''lmql
        """
        {:system}
        You are a senior consultant engaged in a discussion with {self.userName} regarding his/her project idea.
        You and {self.userName} exchanged the following messages up to this point:
        {lastMessagesExchanged}
        """
        if topic.value:
          """
          {:system}
          You also have the following prior information about {topic.description}: {topic.value}
          """
        """
        {:user}
        Does {self.userName} explicitly states that he or she is unwilling to talk about '{topic.description}'? Answer only "yes" or "no".

        {:assistant}
        {self.userName} explicitly stated that he or she is unwilling to talk about '{topic.description}' [UNDISCLOSED]
        """
        UNDISCLOSED = getMostSimilarString(UNDISCLOSED, ["yes", "no"])
        if UNDISCLOSED == "yes":
          return "Undisclosed", False

        if topic.examples:
          """
          {:user}
          The examples must be used solely to provide basis for the formulation of your answer. Dismiss their content and focus on what the user is saying:
          {topic.examples}
          """
        """
        {:user}
        Considering the information provided, extract the information for '{topic.instructions if topic.instructions else topic.description}'.
        {:assistant}
        {topic.instructions if topic.instructions else topic.description}: [INFO]
        """
        """
        {:assistant}
        Considering '{INFO}', answer if more questions would be necessary to completely settle the subject '{topic.description}'. Answer simply 'yes' or 'no': [SATISFACTORY]
        """
        SATISFACTORY = getMostSimilarString(SATISFACTORY, ["yes", "no"])
        if SATISFACTORY == "yes":
            SATISFACTORY = False
        else:
            SATISFACTORY = True
        return INFO.strip(), SATISFACTORY
        '''

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    async def _chooseNextTopic(self, lastMessagesExchanged: str) -> str:  # type: ignore
        '''lmql
        """
        {:user}
        Consider the following messages exchange:
        {lastMessagesExchanged}
        Choose the most appropriate topic from the following json to continue the conversation. Return only the key corresponding to the chosen topic.
        {self._getKeyAndDescriptionOfRemainingTopics()}

        {:assistant}
        The key corresponding to the most appropriate topic is: [KEY]
        """
        return KEY.strip()
        '''

    @lmql.query(model="gpt-4") # TODO possible change line 614 between phases
    # type: ignore
    async def _genMoreTopics(self, numNewTopics: int, lastMessagesExchanged: str) -> Optional[str]:
        '''lmql
        """
        {:system}
        You are a senior consultant guiding {self.userName} in detailing his/her idea. Stay focused on the core issue at hand. 
        You and {self.userName} exchanged the following messages up to this point:
        {lastMessagesExchanged}
        """
        """
        {:system}
        All previous topics were answered with enough detail by {self.userName}.
        The Topics and its answers are as follows:

        {self._getSummaryOfCompleteTopics()}
        """
        """
        Having analyzed the previous exchange, do you think it is necessary to generate new topics to further develop the idea?
        The new topics would be used to extend the conversation in order to get more relevant information in a business point of view and why to implement this idea.
        Continue the conversation in order to advance the idea?
        Answer only "yes" or "no".

        {:assistant}
        It is necessary to generate new topics? [GENERATE]
        """
        GENERATE = getMostSimilarString(GENERATE, ["yes", "no"])
        if GENERATE == "no":
            return None
        else:
            """
            {:system}
            Generate {numNewTopics} new topics to continue the conversation in order to advance the idea.
            Create a list in Python with the following format:

            'key': 'nameOfTheTopic',
            'description': 'Description of the topic'

            Each topic is a dictionary in the list. Write all in english.
            """
            """
            {:assistant}
            {numNewTopics} new topics: [TOPICS]
            """

            return TOPICS
        '''

    @lmql.query(model="gpt-4")
    async def _genBotMessage(self, currentTopicDescription: str, lastMessagesExchanged: str, dbResult: Optional[str] = None, enterSocraticMode: bool = False) -> str: # type: ignore
        '''lmql
        """
        {:system}
        You are a senior consultant guiding {self.userName} in elaborating his/her idea. Keep a direct and objective communication style. Be polite, but avoid expressing gratitude for {self.userName}'s messages, addressing insecurities or offering positive encouragement. Stay focused on the core issue at hand. As a structured consultant, limit each interaction to a single question. Use only the first person singular when referring to yourself. When you identify yourself, use the term "consultor". Conduct the conversation exclusively in Portuguese.
        You two are chatting via instant messages.
        """
        if self.numberOfSuccessfullInteractions == 0: # It is the first user message. Greet him, above all
            """
            {:user}
            Formulate a polite and succint message in {self.language} that welcomes the user, explains your objectives, and talks about {currentTopicDescription}.
            {:assistant}
            Advisor message: [MESSAGE]
            """
            return MESSAGE

        if dbResult:
            """
            {:system}
            Consider the following results retrieved from relevant documents in the database:
            {dbResult}
            """
        if enterSocraticMode:
            """
            {:system}
            {self.userName}'s answer to the previous question was not detailed and did not provide any new information.
            Use the Socratic Method to guide the conversation toward a deeper understanding of the current topic via the following rules:

            Rule 1: Ask open-ended questions to encourage critical thinking;
            Rule 2: Avoid making statements or assertions, focus on asking;
            Rule 3: Seek clarification to ensure a clear understanding of the user's perspective, only when needed;
            Rule 4: Clarify implications and consequences;
            Rule 5: If the user only agrees or disagrees with something, probe for specifics, for instance, "Can you pinpoint what specifically you agree/disagree with?";
            Rule 6: If the user is unsure, narrow down the topic, for instance, "Let's try to explore this together. Can you tell me what aspects you are unsure about?"; 
            Rule 7: Ask for initial impressions, for instance, "Do you have any initial impressions or feelings, even if they're not fully formed?";
            Rule 8: Suggest speculative thinking, for instance, "I understand. What if you had to guess or imagine a possible answer, what might it be?";
            Rule 9: Propose a brainstorm, for instance, "No problem. How about we brainstorm some potential perspectives or ideas together?";
            Rule 10: Encourage any response, for instance, "That's alright. Even if it's just a random thought or feeling, feel free to share it."
            """
        """
        {:user}
        Formulate a polite and concise message with an end question to {self.userName} about '{currentTopicDescription}' in {self.language} that prompts the user to advance in the idea.
        {lastMessagesExchanged}
        Advisor message: [MESSAGE]
        """
        return MESSAGE.strip(" \"`'\n")
        '''

    async def _askBot(self, userMessage: str) -> Tuple[str | globalVariables.RTRN_CODES, bool]:
        enterSocraticMode = False
        numNewTopics = 3
        # self.generatedNewTopics = False  # TODO change to false later

        # lastMessagesExchanged = self._getMessages()
        lastMessagesExchanged = await self._getPartialSummaryOfConversation()

        # sufix = "START"
        # print(f"{sufix:=^100}\n\n")
        # print(lastMessagesExchanged, "\n\n")
        # sufix = "END"
        # print(f"{sufix:=^100}\n\n")
        shouldAbort = abortProcess(userMessage)

        if shouldAbort:
            return globalVariables.RTRN_CODES.ABORT, False

        print("topics: ", self._getKeyAndDescriptionOfRemainingTopics())
        (
            (offensive, cMessage),
            question,
            language,
            keyToNextTopic,

        ) = await asyncio.gather(
            self._isOffensive(message=userMessage),  # type: ignore
            self._isQuestion(message=userMessage),  # type: ignore
            self.getLanguage(message=userMessage),  # type: ignore
            self._chooseNextTopic(lastMessagesExchanged=lastMessagesExchanged),
        )
        
        self._setTopicsSlices(keyToNextTopic)

        self.language = language

        # if isQuestion, # TODO search db

        # keys = []
        # tasks = []
        # for k, t in self.topicsDict.items():
        #     if not ((t.timesAsked > t.maxTimesToAsk) or t.satisfactory):
        #         keys.append(k)
        #         tasks.append(self._getInfoAboutTopic(
        #             topic=t, lastMessagesExchanged=lastMessagesExchanged))

        # results = await asyncio.gather(*tasks)
        # for k, (info, satisfactory) in zip(keys, results):
        #     self.topicsDict[k].satisfactory = self.topicsDict[k].satisfactory or satisfactory
        #     self.topicsDict[k].value = info
        
        info, satisfactory = await self._getInfoAboutTopic(topic=self.topicsDict[self.keyToCurrentTopic], lastMessagesExchanged=lastMessagesExchanged)
        self.topicsDict[self.keyToCurrentTopic].satisfactory = self.topicsDict[self.keyToCurrentTopic].satisfactory or satisfactory
        self.topicsDict[self.keyToCurrentTopic].value = info

        self.keyToCurrentTopic = self._countQuestions(
            keyToNextTopic
        )

        if not self.keyToCurrentTopic:
            return globalVariables.RTRN_CODES.JUST_COMPLETED, False

        isSatisfactory = self.topicsDict[self.keyToCurrentTopic].satisfactory
        print(f"New topics: {not isSatisfactory and not self.isComplete()}")
        if not isSatisfactory and not self.isComplete():
            # if self.getTimesAsked(self.keyToCurrentTopic) > 1:
            if self.getTimesAsked(self.keyToCurrentTopic) > 1 and not enterSocraticMode:
                # Enter socratic mode if the user is not being cooperative, only once per topic
                enterSocraticMode = True
            elif self.getTimesAsked(self.keyToCurrentTopic) > 1 and enterSocraticMode:
                # If the socratic mode is already enabled, explicitly disable it
                enterSocraticMode = False
        else:
            enterSocraticMode = False
            # Check if all topics are complete
            if self.isComplete() and not self.generatedNewTopics:
                newTopics = await self._genMoreTopics(
                    numNewTopics=numNewTopics,
                    lastMessagesExchanged=lastMessagesExchanged,
                )
                # Check if new topics were generated
                self.generatedNewTopics = True
                if newTopics is not None:
                    self.setNewTopics(newTopics)

        botMessage = await self._genBotMessage(
            currentTopicDescription=self.topicsDict[self.keyToCurrentTopic].description,
            enterSocraticMode=enterSocraticMode,
            lastMessagesExchanged=lastMessagesExchanged,
        )

        ####################### + LOGGING #######################
        logger.debug(f"userName      = {self.userName}")
        logger.debug(f"message       = {userMessage}")
        logger.debug(f"offensive     = {offensive}")
        logger.debug(f"question      = {question}")
        logger.debug(f"cMessage      = {cMessage}")
        logger.debug(f"language      = {self.language}")
        logger.debug(f"rawKey        = {keyToNextTopic}")
        logger.debug(f"keyToNextTopic= {self.keyToCurrentTopic}")
        logger.debug(f"satisfactory  = {isSatisfactory}")
        logger.debug(f"socracticMode = {enterSocraticMode}")
        logger.debug(f"shouldAbort   = {shouldAbort}")

        for k, v in self.topicsDict.items():
            logger.debug(f"key={k}")
            logger.debug(f"  description ={v.description}")
            logger.debug(f"  value       ={v.value}")
            logger.debug(f"  satisfactory={v.satisfactory}")
            logger.debug(f"  timesAsked  ={v.timesAsked}")
            logger.debug(f"  undisclosed ={v.undisclosed}")
            logger.debug(f"  complete    ={v.complete()}")
        ####################### - LOGGING #######################
        return botMessage, offensive

    async def askBot(self, userMessage: str) -> Tuple[str | globalVariables.RTRN_CODES, bool]:
        logger.info(
            f"\n{self._getMessages(0)}\n\nSuccessful Interaction with {self.userName} number {self.numberOfSuccessfullInteractions}")
        self.setMessage(type=self.userName, message=userMessage)
        if self.frozen:
            return globalVariables.RTRN_CODES.FROZEN, False

        if self.isComplete():
            return globalVariables.RTRN_CODES.JUST_COMPLETED, False

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            currentTrial = 0
            while currentTrial < self.maxTrials:
                currentTrial += 1
                try:
                    botMessage, offensive = await self._askBot(userMessage)
                    self.numberOfSuccessfullInteractions += 1
                    if type(botMessage) == str:
                        self.setMessage(type="assistant", message=botMessage)
                    return botMessage, offensive
                except Exception as e:
                    logger.error(
                        f"Exception in trial {currentTrial} of {self.maxTrials}",
                        exc_info=True,
                    )

        return globalVariables.RTRN_CODES.ERROR, False

    @lmql.query(max_len=4096, model="gpt-4")
    async def createSummary(self, conversationSummary: str) -> str:  # type: ignore
        '''lmql
        """
        {:system}
        Consider the following:
        - Summary of the conversation:
        {conversationSummary}
        - Conclusions about each topic:
        {self._getSummaryOfCompleteTopics()}
        {:user}
        Create a markdown document summarizing all the information in {self.language};
        Divide the document in subtopics;
        {:assistant}
        summary: [SUMMARY]
        """
        return SUMMARY
        '''
    
    @lmql.query(max_len=8192, model="gpt-4")
    async def createProjectPropose(self, conversationSummary: str, previousDoc: str) -> str:  # type: ignore
        '''lmql
        """
        {:system}
        Consider the following:
        - Summary of the conversation: 
        {conversationSummary}
        - Conclusions about each topic:
        {self._getSummaryOfCompleteTopics()}
        - Previous conversation summary:
        {previousDoc}
        {:user}
        Create a markdown document with all the information in Portuguese;
        The document must follow this template
        1. Idea Owner + Area/Department
        2. Idea Name 
        3. Description of the Idea 
        4. Strategic Objective
        5. Context/Description of the Problem 
        6. Solution Path for the Challenge
        7. Expected Results
        8. Value Levers 
        8. Technologies Involved
        10. Investment Estimate / Budget 
        11. Solution to the Challenge 
        12. Metrication of Results/Goals
        13. Additional Details
        14. Volume (People Impacted)
        15. Effort Estimate (Teams to Involve) 
        16. Macro Schedule
        17. Identified Risks
        18. Submission Date

        I will provide you with 1 optimal example of how this template should be filled. You must attend to and reproduce its format, but the specific content must be based on the conversation. Here is the example:
        1. Idea Owner + Area/Department: Rogério Massi, Logistics and Purchases Departments
        2. Idea Name: IoT-Enabled Supply Chain Optimization
        3. Description of the idea: The idea under development is focused on leveraging IoT technology to transform its supply chain processes. It involves optimizing the end-to-end supply chain, from production facilities to distribution centers, by harnessing the power of real-time data and insights provided by IoT devices. 
        4. Strategic Objective: Accelerate operational efficiency and enhance supply chain resilience to meet growing market demands while ensuring cost-effectiveness.
        5. Context/Description of the Problem: Lack of Real-time Visibility: The company faces challenges in obtaining real-time visibility into its supply chain processes, leading to inefficiencies, delays, and difficulties in proactively managing the production and distribution workflow.\n Ineffective Decision-making: Decision-makers struggle with a lack of timely and accurate data, hindering their ability to make informed decisions, optimize resource allocation, and address potential disruptions in the supply chain.\n Operational Inefficiencies: The existing supply chain processes may suffer from operational inefficiencies, such as equipment downtime and suboptimal inventory management, resulting in increased costs and reduced overall operational effectiveness.
        6. Solution Path for the Challenge: IoT Implementation: Deploy a network of IoT sensors across production lines and distribution centers to collect real-time data on equipment health, inventory levels, and environmental conditions. \n Data Integration: Integrate IoT-generated data with existing enterprise systems, such as ERP and CRM, to enable a holistic view of the supply chain and facilitate data-driven decision-making. \n Predictive Analytics: Implement advanced analytics and machine learning algorithms to predict equipment failures, optimize inventory levels, and identify potential bottlenecks in the supply chain.
        7. Expected Results: Improved Efficiency: Expect a 15% improvement in overall supply chain efficiency through real-time monitoring and predictive maintenance.\n Cost Reduction: Anticipate a 10% reduction in operational costs by optimizing inventory levels, minimizing downtime, and streamlining logistics.\n Enhanced Resilience: Increase supply chain resilience by 20% through early detection of issues and agile response mechanisms enabled by IoT insights.
        8. Value Levers: Operational Excellence: Achieve excellence in supply chain operations through proactive monitoring and streamlined processes.\n Customer Satisfaction: Improve on-time delivery performance, leading to increased customer satisfaction and loyalty. \n Cost Savings: Realize significant cost savings through optimized resource utilization and reduced unplanned downtime.
        9. Technologies Involved: IoT Sensors: Deploy a variety of sensors, including temperature, humidity, and vibration sensors, to monitor conditions in real time.\n Edge Computing: Utilize edge computing to process data locally, reducing latency and ensuring timely decision-making.\n Data Analytics Platforms: Implement robust data analytics platforms to derive actionable insights from the collected IoT data.\n
        10. Investment Estimate / Budget: Approximately $2.5 million for the initial setup of IoT infrastructure, data integration, and implementation of analytics platforms.
        11. Solution to the Challenge: The integration of IoT technology into the supply chain will provide real-time visibility, enabling proactive decision-making, reducing operational costs, and enhancing overall supply chain resilience.
        12. Metrication of Results/Goals: Downtime Reduction: Measure the percentage reduction in unplanned downtime across production facilities. \nInventory Turnover: Track improvements in inventory turnover rates, indicating optimized stock levels. \nResponse Time to Issues: Monitor the time taken to respond and resolve supply chain disruptions, ensuring agility.
        13. Additional Details: Consideration of potential cybersecurity risks associated with IoT implementation and the development of a robust security framework to mitigate such risks.
        14. Volume (People Impacted): Approximately 500 employees involved in production, logistics, and inventory management.
        15. Effort Estimate (Teams to Involve): Cross-functional teams including IoT specialists, data scientists, logistics experts, and IT professionals.
        16. Macro Schedule: A phased approach with an expected implementation timeline of 18 months, including pilot testing and full-scale deployment.
        17. Identified Risks: Potential challenges include data security risks, employee resistance to technology adoption, and interoperability issues with existing systems.
        18. Submission Date: Project proposal to be submitted for review within six weeks to align with strategic planning cycle.
                
        {:assistant}
        document: [SUMMARY]
        """
        return SUMMARY
        '''
    
    @lmql.query(max_len=8192, model="gpt-4")
    async def createProjectCharter(self, conversationSummary: str, previousDoc: str) -> str:  # type: ignore
        '''lmql
        """
        {:system}
        Consider the following:
        - Summary of the conversation: 
        {conversationSummary}
        - Conclusions about each topic:
        {self._getSummaryOfCompleteTopics()}
        - Previous conversation summary:
        {previousDoc}
        {:user}
        Create a markdown document with all the information in Portuguese;
        The document is a Project Charter and should be based on the template below:
        
        Title: provide the project title
        Project Charter
        Date  add the field to input a date

        Prepared by complete with the project manager name

        Project Overview
        Provide a brief description of the project, including its objectives, scope, and expected outcomes.

        Project Objectives
        State the main objectives of the project.
        List any specific goals or milestones to be achieved.

        Stakeholders
        Identify all key stakeholders involved in the project.
        Specify their roles and responsibilities.

        Scope
        Define the boundaries and limitations of the project.
        List what is included and excluded from the project scope.

        Deliverables
        Outline the tangible outcomes or products of the project.
        Include any interim deliverables or milestones.

        Timeline
        Provide an overview of the project schedule, including start and end dates.
        Highlight any critical milestones or deadlines.

        Budget
        Specify the budget allocated for the project.
        Break down the budget into major cost categories.

        Risks and Assumptions
        Identify potential risks that may impact the project's success.
        List any assumptions made during project planning.

        Communication Plan
        Outline how communication will be managed throughout the project.
        Specify the frequency and channels of communication.

        Approval
        List the names of Project manager and sponsor with date for signing
                
        {:assistant}
        document: [SUMMARY]
        """
        return SUMMARY
        '''

    @lmql.query(model="gpt-4")
    def createInitialPath(text, user):
        '''lmql
        'based on the following data {text}'
        'create an initial path to start the idea implementation in markdown in Portuguese'
        "summary: [SUMMARY]"

        return SUMMARY
        '''

    async def summaryUser(self, doc_type="default", previous_doc="") -> str:
        test = True
        nMax = 0
        summary = ""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            while test and nMax < 5:
                nMax += 1
                try:
                    conversarioSummary = await self._summarizeConversation(self.messages, batchSize=6, numSentences=6)
                    if doc_type == "default":
                        summary = await self.createSummary(conversarioSummary)
                    elif doc_type == "Propostas de Projetos":
                        summary = await self.createProjectPropose(conversarioSummary, previous_doc)
                    else:
                        summary = await self.createProjectCharter(conversarioSummary, previous_doc)
                    test = False
                except Exception as e:
                    logging.error(
                        f"error create summary ({nMax} of 5)", exc_info=True)
        return summary


def loadConversation(key: str) -> Conversation:
    """Load a conversation from the database

    Args:

    key (str): The key to load the conversation from

    Returns:

    Conversation: The conversation loaded from the database
    """
    logging.info(f"loading Conversation, key = {key}")
    conversation_data = ConversationDataRepo().findByKey(key)  # type: ignore
    conversation = pickle.loads(conversation_data.data)  # type: ignore
    return conversation

def conversationToJson(conversation: Conversation) -> str:
    jsonDict = {}
    for k, v in conversation.__dict__.items():
        match k:
            case 'topicsDict':
                newJsonDict = {}
                for kt, vt in v.items():
                    newJsonDict[kt] = vt.__dict__
                jsonDict['topicsDict'] = newJsonDict
            case 'messages':
                newJsonList = []
                for m in v:
                    newJsonList.append(m.__dict__)
                jsonDict['messages'] = newJsonList
            case _:
                jsonDict[k] = v

    strJson = json.dumps(jsonDict, indent=2)

    return strJson

def saveConversation(conversation: Conversation, hasData: bool) -> None:
    """Save a conversation to the database

    Args:

    conversation (Conversation): The conversation to save
    key (str): The key to save the conversation under
    hasData (bool): Whether the conversation already exists in the database

    Returns:

    None
    """
    logging.info(f"Saving Conversation, key = {conversation.uniqueId}")
    pickle_str = pickle.dumps(
        obj=conversation, protocol=pickle.HIGHEST_PROTOCOL)
    if hasData:
        ConversationDataRepo().update(key=conversation.uniqueId, # type: ignore
                                      data=pickle_str)  # type: ignore
    else:
        ConversationDataRepo().insert(key=conversation.uniqueId, # type: ignore
                                      data=pickle_str)  # type: ignore


class ChannelMember:
    def __init__(self, userName: str):
        self.contributedToTopic: bool = False
        self.userName: str = userName

class GroupConversation(Conversation):
    def __init__(
        self,
        userName: str,
        userId: str,
        channelUrl: str,
        appId: str,
        uniqueId: str,
        projectTitle: str,
        phaseName: str,
        projectId: str,
        members: list[Tuple[str, str]],
    ):
        super().__init__(
            userName,
            userId,
            channelUrl,
            appId,
            uniqueId,
            projectTitle,
            phaseName,
            projectId,
        )
        self.members = dict([(id, ChannelMember(userName)) for (userName, id) in members])
        self.lastMessageTimestamp = datetime.datetime.now()
        self.keyToCurrentTopic = next(iter(self.topicsDict))

        # self.people = {}
        # for profile in globalVariables.PROFILES:
        #     id = profile["id"]
        #     self.people[id] = ProfileFirstHalf(**profile)

        # self.userName: str = userName
        # self.payloadLoaded = False
        # self.messages = []
        # self.topicsDict: dict[str, TopicForGroup] = {}

        # for dictionay in globalVariables.TOPICS:
        #     key = dictionay["key"]
        #     self.topicsDict[key] = TopicForGroup(**dictionay)

    # def _getMessages(self, numberOfMessages: int = 8, user: bool = True, advisor: bool = True) -> str:
    #     """Returns the n last messages of the conversation

    #     Returns:

    #     str: The messages of the conversation
    #     """
    #     text = ""
    #     for message in self.messages[-numberOfMessages:]:
    #         # if message["type"] == "user" and user:
    #         if message["type"] != "advisor" and user:
    #             sender_id = message["sender_id"]
    #             text += f"\n- {sender_id}_{self.people[id].full_name}: " + message["message"]
    #         if message["type"] == "advisor" and advisor:
    #             text += "\n- Advisor: " + message["message"]

    @lmql.query(model="openai/gpt-3.5-turbo-instruct")
    # type: ignore
    async def _isMessageRelevantToTopic(self, userMessage: str, lastMessagesExchanged: str, topicDescription: str) -> bool:
        '''lmql
        """
        {:system}
        Consider the following:
        - Previous conversation: {lastMessagesExchanged}
        - Message: {userMessage}
        - Current topic: {topicDescription}
        Is the message minimally relevant to the current topic and the conversation in general? answer simply "yes" or "no":
        {:assistant}
        Is the message minimally relevant to the current topic and the conversation in general? [RELEVANT]
        """
        RELEVANT = getMostSimilarString(RELEVANT, ["yes", "no"])
        if RELEVANT == "yes":
            RELEVANT = True
        else:
            RELEVANT = False
        return RELEVANT
        '''

    @lmql.query(model="gpt-4")
    async def _genBotMessage(
        self, 
        currentTopicDescription: str, 
        nextTopicDescription: str,
        lastMessagesExchanged: str,
        dbResult: str | None = None, 
        enterSocraticMode: bool = False,
        greet: bool = False
    ) -> str:  # type: ignore
        '''lmql
        """
        {:system}
        You are an expert consultant guiding a team of employees, via instant messages, in elaborating and implementing their idea.
        Always answer in Portuguese. 
        """
        if greet:
            """
            {:user}
            Introduce yourself and ask them to elaborate on {nextTopicDescription}.
            """
        else:
            """
            {:user}
            Consider the following:
            {lastMessagesExchanged}
            Generate a summary of the conversation, focusing only on the keypoints of ‘{currentTopicDescription}‘. Do not refer to everything that was said in relation to ‘{currentTopicDescription}‘, focus only on the core of ‘{currentTopicDescription}’.
            """
        """
        {:assistant}
        Message to group: [MESSAGE]
        """
        return MESSAGE
        '''

    async def askBot(self, senderId: None | str, userMessage: str) -> Tuple[str | globalVariables.RTRN_CODES, bool]:
        
        if self.isComplete():
            return globalVariables.RTRN_CODES.JUST_COMPLETED, False

        delta = datetime.datetime.now() - self.lastMessageTimestamp
        self.lastMessageTimestamp = datetime.datetime.now()
        
        if self.keyToCurrentTopic:
            topicDescription = self.topicsDict[self.keyToCurrentTopic].description
        else:
            topicDescription = ""

        if not senderId:
            self.numberOfSuccessfullInteractions += 1
            m = await self._genBotMessage(
                currentTopicDescription="", 
                nextTopicDescription=topicDescription,
                lastMessagesExchanged="",
                greet=True
            )
            self.setMessage("Matchias", m)
            return m, False
        
        userName = self.members[senderId].userName
        self.setMessage(userName, userMessage)
        lastMessagesExchanged = self._getMessages()
        
        relevant = await self._isMessageRelevantToTopic(userMessage, lastMessagesExchanged, topicDescription)
        if relevant:
            self.members[senderId].contributedToTopic = True

        contributed = float(len([m for m in self.members.values() if m.contributedToTopic]))
        total = float(len(self.members))
        propOfRelevant = contributed / total
        print(f"relevant={relevant}, propOfRelevant={propOfRelevant}")
        if propOfRelevant > 0.6:# or delta.seconds > 60:
            for m in self.members.values():
                m.contributedToTopic = False

            try:
                topic = self.topicsDict[self.keyToCurrentTopic]
                infoAboutTopic, _ = await self._getInfoAboutTopic(topic, lastMessagesExchanged)
                topic.value = infoAboutTopic
            except:
                print("not possible to get info about topic")

            
            rawKeyToNextTopic = await self._chooseNextTopic(lastMessagesExchanged)
            self.keyToCurrentTopic = self._countQuestions(rawKeyToNextTopic)
            if self.keyToCurrentTopic:
                topicDescription = self.topicsDict[self.keyToCurrentTopic].description
            else:
                topicDescription = ""
            return await self._genBotMessage(
                currentTopicDescription=topicDescription, 
                nextTopicDescription=topicDescription,
                lastMessagesExchanged=lastMessagesExchanged
            ), False
        else:
            return globalVariables.RTRN_CODES.PASS, False
        
def getInfoFromIdeation(conversation: Conversation, description: str, ideasName: str, proposersName: str):
        conversation.topicsDict["description"].value = description
        conversation.topicsDict["description"].satisfactory = True
        conversation.topicsDict["ideasName"].value = ideasName
        conversation.topicsDict["ideasName"].satisfactory = True
        conversation.topicsDict["proposersName"].value = proposersName
        conversation.topicsDict["proposersName"].satisfactory = True
        conversation.payloadLoaded = True

@lmql.query(model="openai/gpt-3.5-turbo-instruct")
async def contextSummary(previousDoc: str) -> str:  # type: ignore
    '''lmql
    """
    {:user}
    Consider the following document:
    {previousDoc}
    Summay the content of the document to create a context to start a conversation.

    {:assistant}
    The summary to start the conversation is: [SUMMARY]
    """
    return SUMMARY
    '''

async def checkSatisfactionFromPayloadValues(conversation: Conversation) -> None:
    """Checks if the information extracted from the payload is satisfactory"""
    keys = []
    tasks = []
    for k, t in conversation.topicsDict.items():
        keys.append(k)
        tasks.append(getInfoAboutTopicFromPayload(conversation, topic=t))

    results = await asyncio.gather(*tasks)
    for k, (info, satisfactory) in zip(keys, results):
        conversation.topicsDict[k].satisfactory = satisfactory
        if satisfactory == True:
            conversation.topicsDict[k].value = info
        else:
            # If value extracted from payload is not satisfactory, ask the user for the information in the future
            conversation.topicsDict[k].value = None

@lmql.query(model="openai/gpt-4")
async def getInfoAboutTopicFromPayload(conversation: Conversation, topic: Topic) -> Tuple[str, bool]: # type: ignore
    '''lmql
    """
    {:system}
    You are an expert advisor chatting with {conversation.userName} about his idea for a project.
    """
    if topic.value:
        """
        {:system}
        You also have the following prior information about {topic.description}: {topic.value}
        """
    if topic.examples:
        """
        {:user}
        Examples:
        {topic.examples}
        """
    """
    {:user}
    Considering the information provided, extract the information for '{topic.instructions if topic.instructions else topic.description}'.
    {:assistant}
    {topic.instructions if topic.instructions else topic.description}: [INFO]
    """
    """
    {:assistant}
    Considering '{INFO}', answer if more questions would be necessary to completely settle the subject '{topic.description}'. Answer simply 'yes' or 'no': [SATISFACTORY]
    """
    SATISFACTORY = getMostSimilarString(SATISFACTORY, ["yes", "no"])
    if SATISFACTORY == "yes":
        SATISFACTORY = False
    else:
        SATISFACTORY = True
    return INFO.strip(), SATISFACTORY
    '''

def abortProcess(message: str) -> bool:
    if "@abort" in message:
        return True
    return False