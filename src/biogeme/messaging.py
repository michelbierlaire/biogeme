"""Obsolete since version 3.11. Standard python logging is now used by Biogeme.

:author: Michel Bierlaire
:date: Tue Mar 21 13:39:14 2023
"""

import logging

logger = logging.getLogger(__name__)


class bioMessage:
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
        logger.warning(
            "The use of messaging.bioMessage is now obsolete. Biogeme "
            "uses the standard logging system from Python. You can obtain "
            "the logger using ``logger = logging.getLogger('biogeme')``"
        )

    def setSilent(self):
        logger.warning(
            "The use of messaging.bioMessage is now obsolete. Biogeme "
            "uses the standard logging system from Python. You can obtain "
            "the logger using ``logger = logging.getLogger('biogeme')``"
        )

    def setWarning(self):
        logger.warning(
            "The use of messaging.bioMessage is now obsolete. Biogeme "
            "uses the standard logging system from Python. You can obtain "
            "the logger using ``logger = logging.getLogger('biogeme')``"
        )

    def setGeneral(self):
        logger.warning(
            "The use of messaging.bioMessage is now obsolete. Biogeme "
            "uses the standard logging system from Python. You can obtain "
            "the logger using ``logger = logging.getLogger('biogeme')``"
        )

    def setDetailed(self):
        logger.warning(
            "The use of messaging.bioMessage is now obsolete. Biogeme "
            "uses the standard logging system from Python. You can obtain "
            "the logger using ``logger = logging.getLogger('biogeme')``"
        )

    def setDebug(self):
        logger.warning(
            "The use of messaging.bioMessage is now obsolete. Biogeme "
            "uses the standard logging system from Python. You can obtain "
            "the logger using ``logger = logging.getLogger('biogeme')``"
        )
