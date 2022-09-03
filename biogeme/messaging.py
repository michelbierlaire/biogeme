"""Singleton managing the various levels of messages

:author: Michel Bierlaire
:date: Mon Jul 22 16:12:00 2019

"""

# Too constraining
# pylint: disable=invalid-name,

import datetime
import biogeme.filenames as bf
import biogeme.version as bv
from biogeme.singleton import Singleton


class bioMessage(metaclass=Singleton):
    """Manages the Biogeme messages"""

    def __init__(self, screenLevel=1):
        """Constructor

        :param screenLevel: level of message that must be displayed on the
            screen:

            - 0: no output (default)
            - 1: warnings only
            - 2: only warnings and general information
            - 3: more verbose
            - 4: debug messages


        :type screenLevel: int
        """
        self.screenLevel = screenLevel  #: screen verbosity level
        self.types = {
            0: 'Silent',
            1: 'Warning',
            2: 'General',
            3: 'Detailed',
            4: 'Debug',
        }  #: names of verbosity levels

        self.resetMessages()

        self.lastLevel = None  #: last level used

    def resetMessages(self):
        """Erase all messages"""
        self.messages = []

    def allMessages(self, screenLevel=None):
        """Report all the messages up to a given level.

        :param screenLevel: level of message that must be reported in the file:

           - 0: no output
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose (default)
           - 4: debug messages

             If None (default), all messages are reported.

        :type screenLevel: int

        :return: all messages.
        :rtype: str.
        """
        result = (
            f'*** Messages from biogeme {bv.getVersion()} [{bv.versionDate}]\n'
        )
        for m, i in self.messages:
            if screenLevel is None or i <= screenLevel:
                result += f'{m}\n'
        return result

    def createLog(self, fileLevel=None, fileName='_biogeme'):
        """Creates a log file

        :param fileLevel: level of message that must be reported in the file:

           - 0: no output
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose (default)
           - 4: debug messages

             If None (default), all messages are reported.
        :type fileLevel: int

        :param fileName: name of the file (without extension).
                         Default: '_biogeme'. A file called _biogeme.log
                         will be created.
        :type fileName: string

        :return: name of the file
        :rtype: str
        """
        completeFileName = bf.getNewFileName(fileName, 'log')
        with open(completeFileName, 'w') as f:
            self.general(f'Log file created: {completeFileName}')
            print(f'*** File created {datetime.datetime.now()} ***', file=f)
            print(
                f'*** Log file from biogeme {bv.getVersion()}'
                f' [{bv.versionDate}]',
                file=f,
            )
            for m, i in self.messages:
                if fileLevel is None or i <= fileLevel:
                    print(m, file=f)
        return completeFileName

    def temporarySilence(self):
        """Temporarily turns off the message, remembering the current
        screen level.
        """
        self.lastLevel = self.screenLevel
        self.screenLevel = 0

    def resume(self):
        """Resume the regular operations of the logger after the use of
        temporarySilence
        """
        if self.lastLevel is not None:
            self.screenLevel = self.lastLevel

    def setSilent(self):
        """Set both screen and file levels to 0"""
        self.screenLevel = 0

    def setWarning(self):
        """Set both screen and file levels to 1"""
        self.screenLevel = 1

    def setGeneral(self):
        """Set both screen and file levels to 2"""
        self.screenLevel = 2

    def setDetailed(self):
        """Set both screen and file levels to 3"""
        self.screenLevel = 3

    def setDebug(self):
        """Set both screen and file levels to 4"""
        self.screenLevel = 4

    def setScreenLevel(self, level):
        """Change the level of messaging for the screen

        :param level: level of message that must be displayed on the screen:

           - 0: no output
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose
           - 4: debug messages


        :type level: int
        """
        self.screenLevel = level

    def addMessage(self, text, level):
        """Add a message

        :param text: text of the message.
        :type text: string

        :param level: level of the message

           - 1: warning
           - 2: general information
           - 3: detailed information
           - 4: debug message

        :type level: int

        :note: adding a message of level 0 is meaningless, as it correspond to
            silentmode.
        """
        theLevel = f'< {self.types[level]} >'
        theMessage = (
            f'[{datetime.datetime.now().strftime("%H:%M:%S")}] '
            f'{theLevel:13} {text}'
        )
        if level != 0:
            self.messages.append((theMessage, level))
            if level <= self.screenLevel:
                print(theMessage)

    def warning(self, text):
        """Add a warning

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text, 1)

    def general(self, text):
        """Add a general message

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text, 2)

    def detailed(self, text):
        """Add a detailed message

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text, 3)

    def debug(self, text):
        """Add a debug message

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text, 4)
