"""Singleton managing the various levels of messages

:author: Michel Bierlaire
:date: Mon Jul 22 16:12:00 2019

"""
import datetime
import biogeme.filenames as bf
import biogeme.version as bv

class Singleton(type):
    """
    A singleton is a class with only one instance
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class bioMessage(metaclass=Singleton):
    """ Manages the Biogeme messages
    """
    def __init__(self,screenLevel=0):
        """ Constructor
        
        :param screenLevel: level of message that must be displayed on the screen:
        
           - 0: no output (default)
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose
           - 4: debug messages


        :type screenlevel: int
        """
        self.screenLevel = screenLevel
        self.types = {0:'Silent',1:'Warning',2:'General',3:'Detailed',4:'Debug'}
        self.messages = []

    def createLog(self,fileLevel=3,fileName="_biogeme"):
        """ Creates a log file

        :param fileLevel: level of message that must be reported in the file:
        
           - 0: no output
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose (default)
           - 4: debug messages

        :type screenlevel: int

        :param fileName: name of the file (without extension). Default: "_biogeme". A file called _biogeme.log will be created.
        :type fileName: string
        """
        completeFileName = bf.getNewFileName(fileName,"log")
        f = open(completeFileName,"w")
        self.general(f"Log file created: {completeFileName}")
        print(f"*** File created {datetime.datetime.now()} ***",file=f)
        print(f"*** Log file from biogeme {bv.getVersion()} [{bv.versionDate}]",file=f)
        for m,i in self.messages:
            if i <= fileLevel:
                print(m,file=f)
        self.messages = []
        return completeFileName

    def setSilent(self):
        """ Set both screen and file levels to 0 """
        self.screenLevel = 0
        self.fileLevel = 0

    def setWarning(self):
        """ Set both screen and file levels to 1 """
        self.screenLevel = 1
        self.fileLevel = 1
        
    def setGeneral(self):
        """ Set both screen and file levels to 2 """
        self.screenLevel = 2
        self.fileLevel = 2
        
    def setDetailed(self):
        """ Set both screen and file levels to 3 """
        self.screenLevel = 3
        self.fileLevel = 3

    def setDebug(self):
        """ Set both screen and file levels to 4 """
        self.screenLevel = 4
        self.fileLevel = 4
        
    def setScreenLevel(self,level):
        """ Change the level of messaging for the screen

        :param level: level of message that must be displayed on the screen:
        
           - 0: no output
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose
           - 4: debug messages


        :type level: int
        """
        self.screenLevel = level

    def setFileLevel(self,level):
        """ Change the level of messaging for the file

        :param level: level of message that must be reported in the file:
        
           - 0: no output
           - 1: warnings only
           - 2: only warnings and general information
           - 3: more verbose
           - 4: debug messages

        :type level: int
        """
        self.fileLevel = level
        
    def addMessage(self,text,level):
        """ Add a message

        :param text: text of the message.
        :type text: string

        :param level: level of the message

           - 1: warning
           - 2: general information
           - 3: detailed information
           - 4: debug message

        :note: adding a message of level 0 is meaningless, as it correspond to silentmode.
        """
        theLevel = f"< {self.types[level]} >"
        theMessage = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {theLevel:13} {text}"
        if level != 0:
            self.messages.append((theMessage,level))
            if level <= self.screenLevel:
                print(theMessage)

    def warning(self,text):
        """ Add a warning

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text,1)

    def general(self,text):
        """ Add a general message

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text,2)

    def detailed(self,text):
        """ Add a detailed message

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text,3)

    def debug(self,text):
        """ Add a debug message

        :param text: text of the message.
        :type text: string

        """
        self.addMessage(text,4)
        
    

    
