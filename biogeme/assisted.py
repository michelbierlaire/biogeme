"""File assisted.py
:author: Michel Bierlaire, EPFL
:date: Thu Sep 17 16:21:01 2020

Assisted specification for choice models
"""

# pylint: disable=too-many-lines, invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=too-many-public-methods, too-many-branches
# pylint: disable=too-many-statements

from collections import namedtuple
import random
import copy
import html
import biogeme.biogeme as bio
import biogeme.messaging as msg
import biogeme.exceptions as excep
from biogeme import vns
from biogeme.expressions import Beta, bioMultSum

# This tuple is imported here, so that it can be imported from here later on.
from biogeme.segmentation import DiscreteSegmentationTuple, segment_parameter

logger = msg.bioMessage()


TermTuple = namedtuple('TermTuple', 'attribute segmentation bounds validity')
SegmentedParameterTuple = namedtuple(
    'SegmentedParameterTuple', 'dict combinatorial'
)


class variable:
    """
    Class representing the possible specifications of a variable
    """

    def __init__(self, name, expression):
        """
        :param name: name of the variable
        :type name: str

        :param expression: Biogeme expression of the variable.
        :type expression: :class:`biogeme.expressions.Expression`
        """

        self.name = name  #: name of the variable

        self.expression = expression  #: Biogeme expression for the variable

        self.active = False  #: True if variable is active

        self.generic = False  #: True if the variable is generic.

        self.genericName = None  #: Name of the generic variable

        self.nonlinearSpec = None
        """ Function with the nonlinear specifications.
        """

        self.used = False  #: True if variable used in the model.

    def __str__(self):
        """
        Print the specification of the variable
        """
        if not self.active:
            return f'{self.name} [deactivated]'

        if self.nonlinearSpec is None:
            if self.generic:
                return f'{self.name} [generic]'
            return f'{self.name} [alt. spec.]'

        if self.generic:
            return (
                f'{self.name}_{self.nonlinearSpec(self.expression)[0]}'
                f' [generic]'
            )

        return (
            f'{self.name}_{self.nonlinearSpec(self.expression)[0]}'
            f' [alt. spec.]'
        )

    def makeGeneric(self, yes, name):
        """A variable can have two status: generic or alternative
        specific. This function changes the status

        :param yes: if True, status is set to "generic".
                    If False, status is set to "alt. specific".
        :type yes: bool

        :param name: name of the generic version of the variable
        :type name: str

        """
        self.generic = yes
        self.genericName = name

    def getExpression(self):
        """Returns the biogeme expression of the specification of the variable.

        :return: expression accozunting for the status and the
            nonlinear specification.
        :rtype: :class:`biogeme.expressions.Expression`

        """
        if not self.active:
            return None
        if self.nonlinearSpec is None:
            return self.expression
        return self.nonlinearSpec(self.expression)[1]


class groupOfVariables:
    """Class representing groups of variables. All variables in the group
    will have the same nonlinear spec. They can also share the same
    coefficient.

    """

    def __init__(self, name, variables, nonlinearSpecs):
        """
        Ctor

        :param name: name of the group of variables
        :type name: str

        :param variables: list of variables in the group
        :type varariables: list(variable)

        :param nonlinearSpecs: list of possible nonlinear specifications
        :type nonlinearSpecs: list(function)
        """
        self.name = name  #: name of the group of variables.

        self.variables = variables  #: list of variables in the group.

        self.genericForbiden = len(self.variables) <= 1
        """ True of the group cannot be made generic
        """

        self.generic = not self.genericForbiden
        """ True if the group is generic.
        """

        self.alwaysActive = False
        """ True if the group is always active.
        """

        self.active = False  #: True if the group is active.

        self.nonlinearSpecs = nonlinearSpecs
        """ list of possible nonlinear specifications
        """

        self.selection = 0
        """Index of the selected non linear specification.
        """

        self.linear = True  #: True if linear specification.

    def __str__(self):
        v = [s.__str__() for s in self.variables]
        if self.generic:
            return f'{self.name}: {v}  [generic][active:{self.active}]'

        return f'{self.name}: {v} [not generic][active:{self.active}]'

    def forbidGeneric(self):
        """Forbid the generic specification in the group"""
        self.genericForbiden = True
        self.generic = False

    def forceActive(self):
        """Force the variables in the group to be active."""
        logger.detailed(
            f'Group of variables {self.name} ' f'must be in the model.'
        )
        self.activate(True)
        self.alwaysActive = True

    def activate(self, yes):
        """A group of variables can have two status: activated or not. This
        function changes the status.

        :param yes: if True, activates the group. If False,
            desactivate the group.
        :type yes: bool

        """
        if self.alwaysActive:
            return

        self.active = yes
        for v in self.variables:
            v.active = yes

    def swapActivate(self):
        """Change the activation status"""
        if self.alwaysActive:
            raise excep.biogemeError(
                f'Variable {self.name} ' f'cannot be made inactive.'
            )
        return self.activate(not self.active)

    def makeGeneric(self, yes):
        """
        A group of variables can be generic or alternative specific.
        This function changes the status.

        :param yes: if True, status is set to "generic".
                    If False, status is set to "alt. specific".
        :type yes: bool

        :raise biogemeError: if the variable cannot be made generic.
        """
        if yes and self.genericForbiden:
            raise excep.biogemeError(
                f'Variable {self.name} ' f'cannot be made generic.'
            )

        self.generic = yes
        for v in self.variables:
            v.makeGeneric(yes, self.name)

    def swapGeneric(self):
        """Change the generic/alt. specific status"""
        if self.genericForbiden:
            raise excep.biogemeError(
                f'Variable {self.name} ' f'cannot be made generic.'
            )

        return self.makeGeneric(not self.generic)

    def setLinear(self, yes):
        """A group of variables can be linear or not. This function changes
        the status.

        :param yes: if True, status is set to "linear".
                    If False, status is set to "nonlinear".
        :type yes: bool

        """
        if self.nonlinearSpecs is None:
            return
        self.linear = yes
        for v in self.variables:
            if yes:
                v.nonlinearSpec = None
            else:
                v.nonlinearSpec = self.nonlinearSpecs[self.selection]

    def swapLinear(self):
        """Change the linearity status."""
        if self.nonlinearSpecs is None:
            return
        self.setLinear(not self.linear)

    def setSelection(self, sel):
        """Set the selection of the nonlinear specification.

        :param sel: index of list self.nonlinearSpecs corresponding to
            the nonlinear spec.
        :type sel: int

        :raise biogemeError: if the index is out of range.

        """
        if self.nonlinearSpecs is None:
            return

        if sel < 0 or sel >= len(self.nonlinearSpecs):
            raise excep.biogemeError(
                f'Index {sel} out of range [0, '
                f'{len(self.nonlinearSpecs)-1}]'
            )
        self.selection = sel
        if not self.linear:
            for v in self.variables:
                v.nonlinearSpec = self.nonlinearSpecs[self.selection]

    def getDecisions(self):
        """The decision is an integer representing the decisions with
        respect to the group of variables:


        :return: decision with respect to the group of variables

        - -3 if it is inactive
        - -2 if it is active, generic and linear
        - -1 if it is active, alt. specific and linear
        - index of the nonlinear specification if active, generic
          and nonlinear.
        - 100 plus the index of the nonlinear specification if active,
          alt. specific and nonlinear.

        :rtype: int
        """
        if not self.active:
            result = -3
        elif self.linear and self.generic:
            result = -2
        elif self.linear and not self.generic:
            result = -1
        elif self.generic:
            result = self.selection
        else:
            result = 100 + self.selection

        return result

    def setDecisions(self, decision):
        """
        Implement the decision, after verifying its validity

        :param decision:

        - -3 if it is inactive
        - -2 if it is active, generic and linear
        - -1 if it is active, alt. specific and linear
        - index of the nonlinear specification if active, generic
          and nonlinear.
        - 100 plus the index of the nonlinear specification if active,
          alt. specific and nonlinear.

        :type decision: int

        :raise biogemeError: if the decision is to deactive, while
            the group should always be active.
        """
        if decision == -3 and self.alwaysActive:
            error_msg = (
                f'Group of variables {self.name} cannot be desactivated'
            )
            raise excep.biogemeError(error_msg)
        self.activate(decision != -3)
        if decision == -3:
            return
        if decision == -2:
            self.makeGeneric(True)
            self.setLinear(True)
            return
        if decision == -1:
            self.makeGeneric(False)
            self.setLinear(True)
            return
        self.setLinear(False)
        if self.nonlinearSpecs is None:
            raise excep.biogemeError(
                f'No nonlinear specification has '
                f'been provided for {self.name}'
            )

        if decision < 100:
            self.makeGeneric(True)
            self.setSelection(decision)
            return
        self.makeGeneric(False)
        self.setSelection(decision - 100)
        return


class term:
    """Class representing the possible specifications of one term of the
    utility function

    """

    def __init__(self, var, aSegmentation, bounds, validity):
        """
        Ctor

        :param var: variable of the term.
        :type var: variable

        :param aSegmentation: discrete segmentation for the parameter
        :type aSegmentation: segmentation

        :param bounds: bounds on the coefficient
        :type bounds: tuple(float, float)

        :param validity: function checking the validity of the coefficient.
        :type validity: bool f(float)

        """
        self.var = var  #: variable of the term

        self.segmentation = aSegmentation
        """ discrete segmentation of the parameter
        """

        self.coef_names = None
        """names of the Beta parameters involved in the specification of the
        term.
        """

        self.validity = validity
        """ function checking the validity of the coefficient.
        """

        self.bounds = bounds  #: bounds on the coefficient

        if self.var is not None:
            self.var.used = True

        if self.segmentation is not None:
            self.segmentation.used = True

    def getDecisions(self):
        """The decision is a dict, where the keys are the name of the
        socioeconomic variables, and the value are a boolean mentioning if
        it is active or not.

        :return: decision of activation of the socio-eco variables for
            the segmentation.
        :rtype: dict(str: bool)

        """
        if self.segmentation is None:
            return None
        return self.segmentation.getDecisions()

    def setDecisions(self, decisions):
        """Implement the specification decisions, represented as a dict,
        where the keys are the name of the socioeconomic variables,
        and the value are a boolean mentioning if it is active or not.

        :param decisions: decision of activation of the socio-eco variables for
            the segmentation.
        :type decisions: dict(str: bool)

        """
        if self.segmentation is None:
            return
        self.segmentation.setDecisions(decisions)

    def getBeta(self, altname):
        """
        Obtain the name of the coefficient of the term

        :param altname: name of the alternative
        :type altname: str

        :return: name of the coefficient
        :rtype: str
        """
        if self.var is None:
            return f'beta_{altname}'
        if self.var.generic:
            return f'beta_{self.var.genericName}'

        return f'beta_{self.var.name}_{altname}'

    def isValid(self, altname, estimationResults):
        """
        Check the validity of the estimated coefficient

        :param altname: name of the laternative
        :type altname: str

        :param estimationResults: results of the estimation with Biogeme
        :type estimationResults: :class:`biogeme.results.bioResults`

        :return: True if the valus is valid, False otherwise.
        :rtype: bool

        :raise biogemeError: if the parameter has not be estimated.
        """
        if self.validity is None:
            return True, None
        if not self.var.active:
            return True, None
        estimatedValues = estimationResults.getBetaValues()
        status_msg = None
        for the_beta in self.coef_names:
            print(f'Check validity of {the_beta=}')
            val = estimatedValues.get(the_beta)
            if val is None:
                p = list(estimatedValues.keys())
                raise excep.biogemeError(
                    f'Parameter {b}'
                    f' has not been estimated. '
                    f'Estimated parameters: '
                    f'{p}'
                )

            if not self.validity(val):
                if status_msg is None:
                    status_msg = 'Invalid parameter(s):'
                status_msg += f' {the_beta} in alternative {altname}: {val}'

        return status_msg is None, status_msg

    def getExpression(self, altname):
        """
        Build the Biogeme expression for the term

        :param altname: name of the alternative
        :type altname: str

        :return: eypression for the term
        :rtype: :class:`biogeme.expressions.Expression`
        """
        if self.var is None:
            coef_name = self.getBeta(altname)
            if self.segmentation is None:
                return Beta(coef_name, 0, self.bounds[0], self.bounds[1], 0)

            self.coef_names = [coef_name]
            return self.segmentation.getExpression(coef_name, self.bounds)

        if not self.var.active:
            self.coef_names = None
            return None

        coef_name = self.getBeta(altname)
        if self.segmentation is None:
            self.coef_names = [coef_name]
            return self.var.getExpression() * Beta(
                coef_name, 0, self.bounds[0], self.bounds[1], 0
            )

        theBeta = self.segmentation.getExpression(coef_name, self.bounds)
        self.coef_names = list(theBeta.setOfBetas())
        return self.var.getExpression() * theBeta

    def describe(self):
        """
        Provides a short description of the term.

        :return: short description.
        :rtype: str
        """
        if self.var is None:
            result = 'Cte.'
        elif not self.var.active:
            return ''
        else:
            result = str(self.var)

        if self.segmentation is not None:
            result += ' '
            result += self.segmentation.describe()
        return result


class utility:
    """Class representing the possible specifications of a utility function"""

    def __init__(self, alternativeId, name, terms):
        """
        Ctor.

        :param alternativeId: id of the alternative.
        :type alternativeId: int

        :param name: name of the alternative
        :type name: str

        :param terms: terms of the utility function.
        :type terms: list(term)

        """
        self.name = name  #: name of the alternative

        self.id = alternativeId  #: id of the alternative

        self.terms = terms  #: list of terms in the utility function

    def getExpression(self):
        """
        Obtain the Biogeme expression for the utility function.

        :return: Biogeme expression
        :rtype: :class:`biogeme.expressions.Expression`
        """
        theTerms = [
            t.getExpression(self.name)
            for t in self.terms
            if t.getExpression(self.name) is not None
        ]
        return bioMultSum(theTerms)


class socioEconomic:
    """Class representing  socio-economic characteristic"""

    def __init__(self, name, expression, values):
        """Ctor

        :param name: name of the segmentation variable
        :type name: str

        :param expression: Biogeme expression of the variable
        :type expression: meth:`biogeme.expressions.Expression`

        :param values: dict with values that it can take as keys, and a name
            describing them as values.
        :type values: dict(int: str)

        """
        self.name = name  #: name of the segmentation variable

        self.expression = expression  #: Biogeme expression of the variable

        self.values = values
        """dict with values that it can take as keys, and a name
        describing them as values.
        """

        self.active = False  #: True if the segmentation variable is active.

    def combine(self, existingValues):
        """Generates the possible combinations of values,
        corresponding to segments.
        """
        if existingValues is None:
            return [
                ([self.expression], [k], [v]) for k, v in self.values.items()
            ]

        combination = []
        for k, v in self.values.items():
            for triplet in existingValues:
                combination.append(
                    (
                        triplet[0] + [self.expression],
                        triplet[1] + [k],
                        triplet[2] + [v],
                    )
                )
        return combination


class segmentation:
    """Class representing the possible segmentations"""

    def __init__(self, name, dictOfSocioEco, combinatorial):
        """
        Ctor
        """
        self.dictOfSocioEco = {
            k: socioEconomic(k, v[0], v[1]) for k, v in dictOfSocioEco.items()
        }
        """dict of object of class socioEconomic characterizing the
        segmentation.
        """

        self.combinatorial = combinatorial
        """True if all combinations are considered"""

        self.listOfVariables = []  #: list of variables involved

        self.alwaysActive = False  #: True if it must always be active

        self.name = name  #: name of the segmentation

        self.used = False  #: True if used.

    def __str__(self):
        return f'{self.dictOfSocioEco}'

    def getDecisions(self):
        """The decision is a dict, where the keys are the name of the
        socioeconomic variables, and the value are a boolean mentioning
        if it is active or not.

        :return: decision of activation of the socio-eco variables for
            the segmentation.
        :rtype: dict(str: bool)

        """
        decision = {k: v.active for k, v in self.dictOfSocioEco.items()}
        return decision

    def setDecisions(self, decisions):
        """Implement the specification decisions, represented as a dict,
        where the keys are the name of the socioeconomic variables,
        and the value are a boolean mentioning if it is active or not.

        :param decisions: decision of activation of the socio-eco variables for
            the segmentation.
        :type decisions: dict(str: bool)

        :raise biogemeError: if the name of a segmentation is unknown
        """
        for k, v in decisions.items():
            try:
                self.dictOfSocioEco[k].active = v
            except KeyError as e:
                error_msg = (
                    f'Key {k} is unknown. '
                    f'Available segmentations: '
                    f'{list(self.dictOfSocioEco.keys())}'
                )
                raise excep.biogemeError(error_msg) from e

    def isActive(self):
        """Check if there are active variables for this segmentation."""
        if self.alwaysActive:
            return True
        activeVariables = sum(
            v.active if v is not None else True for v in self.listOfVariables
        )
        return activeVariables > 0

    def getBetaNames(self, coef_name):
        """Get the name of the parameters for all combinations."""

        expr = self.getExpression(coef_name, (None, None))
        betas = expr.setOfBetas()
        return [betas]
        combinations = None
        for v in self.dictOfSocioEco.values():
            if v.active:
                combinations = v.combine(combinations)

        if combinations is None:
            return [coef_name]

        betas = []
        for triplet in combinations:
            theCoefName = coef_name
            for _, _, name in zip(triplet[0], triplet[1], triplet[2]):
                theCoefName = f'{theCoefName}_{name}'
            betas.append(theCoefName)
        return betas

    def getExpression(self, coef_name, bounds):
        """Obtain the Biogeme expression.

        :param coef_name: name of the coefficient
        :type coef_name: str

        :param bounds: bounds on the coefficient
        :type bounds: tuple(float, float)

        :return: biogeme  expression for the segmentation
        :rtype: :class:`biogeme.expressions.bioMultSum`
        """

        #        combinations = None
        #        for v in self.dictOfSocioEco.values():
        #            if v.active:
        #                combinations = v.combine(combinations)
        #
        #        if combinations is None:
        #            return Beta(coef_name, 0, bounds[0], bounds[1], 0)
        #
        #        listOfTerms = []
        #        for triplet in combinations:
        #            theCoefName = coef_name
        #            listOfConditions = []
        #            for var, value, name in zip(triplet[0], triplet[1], triplet[2]):
        #                theCoefName = f'{theCoefName}_{name}'
        #                listOfConditions.append(var == value)
        #            aTerm = Beta(theCoefName, 0, bounds[0], bounds[1], 0)
        #            for t in listOfConditions:
        #                aTerm = aTerm * t
        #            listOfTerms.append(aTerm)
        #        before =  bioMultSum(listOfTerms)

        the_coef = Beta(coef_name, 0, bounds[0], bounds[1], 0)
        list_of_segmentations = [
            DiscreteSegmentationTuple(variable=v.expression, mapping=v.values)
            for v in self.dictOfSocioEco.values()
            if v.active
        ]
        return segment_parameter(
            the_coef, list_of_segmentations, self.combinatorial
        )

    def describe(self):
        """Description of the segmentation

        :return: description
        :rtype: str
        """
        active = [t.name for t in self.dictOfSocioEco.values() if t.active]
        if active:
            return '<' + ', '.join(active) + '>'
        return ''


class specificationProblem(vns.problemClass):
    """Class defining the choice model specification problem"""

    def __init__(
        self,
        name,
        database,
        theVariables,
        theGroups,
        genericForbiden,
        forceActive,
        theNonlinearSpecs,
        theSegmentations,
        utilities,
        availabilities,
        choice,
        models,
    ):
        """Ctor.

        :param name: name of the problem.
        :type name: str

        :param database: data for the estimation
        :type database: :class:`biogeme.database.Database`

        :param theVariables: variables involved in the model and their names
        :type theVariables: dict(str: :class:`biogeme.expressions.Expression`)

        :param theGroups: variables in the same groups share the same
            transforms and activation status. Each group is characterized
            by its name, and is associated to a list of variables,
            identified by their name.
        :type theGroups: dict(str: list(str))

        :param genericForbiden: groups of variables that must be
            alternative specific.

        :type genericForbiden: list(str)

        :param forceActive: groups of variables that must be in the model.
        :type forceActive: list(str)

        :param theNonlinearSpecs: associates a group of variables or a
            variable with a list of possible nonlinear
            transformations. Each transformation is a function that takes
            one argument (the variable), and return a tuple with

            - the name of the nonlinear transform
            - the expression of the transform.

             Examples of such a function::

                def sqrt(x):
                    return 'sqrt', x**0.5

                def boxcox(x):
                    ell = Beta(f'lambda', 1, 0.0001, 3.0, 0)
                    return 'Box-Cox', models.boxcox(x, ell)

        :type theNonlinearSpecs: dict(str: list( fct() ))


        :param theSegmentations: a dictionary, with keys being names
            and values beeing tuples (var, segments), where

            - var is the name of the variable
            - segments is a dict with keys being the value of the variable
              characterizing a segment, and the value being the name of
              the segment.

            Example::

                {'Income': (Income, {1: '<2500',
                                     2: '2051_4000',
                                     3: '4001_6000',
                                     4: '6001_8000',
                                     5: '8001_10000',
                                     6: '>10000',
                                     -1: 'unknown'}),
                 'Gender': (Gender, {1: 'male',
                                     2: 'female',
                                    -1: 'unkown'}),

        :type theSegmentations: dict(str, tuple(biogeme.expression.Expression,
                                                dict(int, str)))

        :param utilities: specification of the utility functions. It
            is a dict where

            - the keys are the ID of the alternatives.

            - the values are a tuple containing the name of the
              alternative and the specification.

            The specification is a list of terms. A term is a tuple with
            the name of the variable, the name of the segmentation, the
            bounds on the coeffcient, and a function checking the validity
            of the corresponding parameter (typically, check its
            sign). All can be None. If they are all None, it corresponds
            to the alternative specification constant, without any
            segmentation and any assumption on the sign.

            Example::

                utility_pt = [('PT cte', 'Seg. cte', (None, None), None),
                  ('PT travel time', 'Seg. time', (None, 0), None),
                  ('PT travel cost', 'Seg. PT cost', (None, None), isNegative),
                  ('PT Waiting time', 'Seg. wait', (None, 0), None)]


                utility_car = [('Car cte', 'Seg. cte', (None, None), None),
                  ('Car travel time', 'Seg. time', (None, 0), None),
                  (
                   'Car travel cost',
                   'Seg. car cost',
                   (None, None),
                   isNegative
                  ),
                  ('Nbr of cars', 'Seg nbr cars', (None, None), None)]

                utility_sm = [(
                               'Distance',
                               'Seg. dist',
                               (None, None),
                               isNegative
                              )]

                choiceModel = {0: ('pt', utility_pt),
                               1: ('car', utility_car),
                               2: ('sm', utility_sm)}

        :type utilities: dict(int, tuple(str,
                         list(tuple(str, str, tuple(float, float),function))))

        :param availabilities: dict describing the availability of the
            alternatives.

        :type availabilities:
            dict(int, :class:`biogeme.expressions.Expression`)

        :param choice: expression for the observed choice
        :type choice: :class:`biogeme.expressions.Expression`

        :param models: dict of possible models. A model is a function
            that takes the utilities and the availabilities, and return
            the loglikelihood expression.

        :type models: dict(str, fct)

        :raise biogemeError: if a variable is found in two different groups.

        :raise biogemeError: if the list of groups forbiden to be
            active contains an unknown group.

        :raise biogemeError: if the list of groups forced to be active
            contains an unknown group.

        :raise biogemeError: if some variables are not in any group.

        :raise biogemeError: if a segmentation in a utility function is
            unknown.

        :raise biogemeError: if a variable in a utility function is unknown.

        :raise biogemeError: if some variables are not used.



        """

        self.archive = {}
        """Dictionary, where the keys are solutions (objects of type
        :class:`biogeme.vns.solutionClass`) and the values are the
        estimation results (objects of type
        :class:`biogeme.results.bioResults`).
        """

        self.name = name  #: name of the problem.

        self.database = database
        """object of type :class:`biogeme.database.Database`, containing the
        data.
        """

        # First check if all the variables are in a group. For those
        # who are not, create a group with a single variable
        groupForVar = {k: None for k in theVariables}
        for group, listOfVars in theGroups.items():
            for x in listOfVars:
                if groupForVar[x] is None:
                    groupForVar[x] = group
                else:
                    error_msg = (
                        f'Var {x} cannot be both in group {group}'
                        f' and group {groupForVar[x]}'
                    )
                    raise excep.biogemeError(error_msg)

        for x, g in groupForVar.items():
            if g is None:
                logger.detailed(f'Variable {x} is alone in a group.')
                theGroups[x] = [x]

        self.theVariables = {
            k: variable(k, v) for k, v in theVariables.items()
        }
        """dict of variables, where the keys are the names, and the values are
        objects of class ``variable``
        """

        self.theGroups = {
            k: groupOfVariables(
                k, [self.theVariables[i] for i in v], theNonlinearSpecs.get(k)
            )
            for k, v in theGroups.items()
        }
        """dict of groups of variables, where the keys are the names, and the
        values are objects of class ``groupOfVariables``
        """

        if genericForbiden is not None:
            for k in genericForbiden:
                try:
                    self.theGroups[k].forbidGeneric()
                except KeyError as e:
                    error_msg = (
                        f'Unknown group of variables {k} '
                        f'in the list {genericForbiden}'
                    )
                    raise excep.biogemeError(error_msg) from e

        if forceActive is not None:
            for k in forceActive:
                try:
                    self.theGroups[k].forceActive()
                except KeyError as e:
                    error_msg = (
                        f'Unknown group of variables {k} '
                        f'in the list {forceActive}'
                    )
                    raise excep.biogemeError(error_msg) from e

        check = {v: False for v in theVariables}
        for k, myvars in theGroups.items():
            for v in myvars:
                check[v] = True
        for k, c in check.items():
            if not c:
                error_msg = (
                    f'Variables not in any group: '
                    f'{[k for k, c in check.items() if not c]}'
                )
                raise excep.biogemeError(error_msg)

        self.theSegmentations = {
            k: segmentation(k, v.dict, v.combinatorial)
            for k, v in theSegmentations.items()
        }
        """dict of segmentations, where the keys are the names, and the
        values are objects of class ``segmentation``
        """

        self.utilities = utilities
        """specification of the utility functions. See
        :class:`biogeme.assisted.specificationProblem.__init__`
        """

        self.choice = choice  #: expression for the observed choice

        self.availability = availabilities
        """dict describing the availability of the alternatives.
        """

        self.theAlternatives = {}
        """Dict of utility functions, where the keys are the id of the
        alternatives, and the values are objects of type
        :class:`biogeme.assisted.utility`

        """

        # self.models = [(k, v) for k, v in models.items()]

        self.models = list(models.items())
        """ List tuple (name, model).  A model is a function
        that takes the utilities and the availabilities, and return
        the loglikelihood expression.
        """

        self.selectedModel = 0  #: index of the selected model

        self.maximumNumberOfParameters = 200
        """maximum number of parameters allowed in a specification. If the
        current model has more parameters, it is declared invalid and
        rejected by the algorithm.
        """

        self.lastOperator = None  #: Last operator used

        self.operators = {
            'Change segmentation': self.changeSegmentation,
            'Increase segmentation': self.increaseSegmentation,
            'Decrease segmentation': self.decreaseSegmentation,
            'Change linearity': self.changeLinearity,
            'Change variables': self.changeVariables,
            'Change generic': self.changeGeneric,
            'Change nonlinearity': self.changeNonlinearity,
            'Change model': self.changeModel,
        }
        """Dict of operators, where the keys are the names, and the values are
        the function imple,enting the operators.
        """

        self.operatorsManagement = vns.operatorsManagement(
            self.operators.keys()
        )
        """ Object of type :class:`biogeme.vns.operatorsManagement`
        """

        # Check the consistency of the input
        for k, u in self.utilities.items():
            # First check that the tuples have the correct length
            for t in u[1]:
                if len(t) != 4:
                    err_msg = f'tuple must contain 4 elements: {t}'
                    raise excep.biogemeError(err_msg)
            # All segmentations must exist
            for t in u[1]:
                # t[0] name of the attribute
                # t[1] name of the segmentation
                # t[2] bounds
                # t[3] function checking validity
                if (
                    t.segmentation is not None
                    and self.theSegmentations.get(t.segmentation) is None
                ):
                    raise excep.biogemeError(
                        f'Segmentation {t.segmentation} ' f'does not exist'
                    )
                if (
                    t.attribute is not None
                    and self.theVariables.get(t.attribute) is None
                ):
                    raise excep.biogemeError(
                        f'Attribute {t.attribute} does not exist'
                    )
                self.theAlternatives[k] = utility(
                    k,
                    u[0],
                    [
                        term(
                            self.theVariables.get(tt.attribute),
                            self.theSegmentations.get(tt.segmentation),
                            tt.bounds,
                            tt.validity,
                        )
                        for tt in u[1]
                    ],
                )
                if t.segmentation is not None:
                    self.theSegmentations[
                        t.segmentation
                    ].listOfVariables.append(
                        self.theVariables.get(t.attribute)
                    )
                # If the segmentation is not associated with a
                # variable, it must always be active
                if t.attribute is None:
                    self.theSegmentations[t.segmentation].alwaysActive = True

        self.decisions = self.getDecisions()
        """
        The decisions consist of:

        - a dict of decisions for each group of variables.
        - for each utility, a list of decisions for each term.
        - the selected model

        Type: tuple(dict(str: int), dict(str: list(dict(str: bool))), int)
        """

        unused = []
        for v in self.theVariables.values():
            if not v.used:
                unused.append(v.name)
        if unused:
            raise excep.biogemeError(
                f'The following variables ' f'are not used: {unused}'
            )
        unused = []
        for s in self.theSegmentations.values():
            if not s.used:
                unused.append(s.name)
        if unused:
            raise excep.biogemeError(
                f'The following segmentations ' f'are not used: {unused}'
            )

    def getBiogemeModel(self):
        """Build the Biogeme expressions of a given specification"""
        V = {k: v.getExpression() for k, v in self.theAlternatives.items()}
        logprob = self.models[self.selectedModel][1](
            V, self.availability, self.choice
        )
        b = bio.BIOGEME(
            self.database,
            logprob,
            suggestScales=False,
            userNotes=self.describeHtml(),
        )
        b.generateHtml = False
        b.generatePickle = False
        return b

    def getDecisions(self):
        """
        The decisions consist of:

        - a dict of decisions for each group of variables.
        - for each utility, a list of decisions for each term.
        - the selected model

        :return: all decisions
        :rtype: tuple(dict(str: int), dict(str: list(dict(str: bool))), int)
        """

        groupDecisions = {
            k: v.getDecisions() for k, v in self.theGroups.items()
        }
        termDecisions = {
            k: [t.getDecisions() for t in u.terms]
            for k, u in self.theAlternatives.items()
        }
        return groupDecisions, termDecisions, self.selectedModel

    def setDecisions(self, decisions):
        """Implement the specification decisions

        :param decisions: specification decisions
        :type decisions: tuple(dict(str: int),
                               dict(str: list(dict(str: bool))),
                               int)

        :raise biogemeError: if a group of variables is unknown.

        :raise biogemeError: if an alternative is unknown.

        :raise biogemeError: if the length of decisions is
            inconsistent with the number of terms in the utility function.

        :raise biogemeError: if the njumber of the model is out of range.

        """
        if self.decisions == decisions:
            return

        self.decisions = decisions
        groupDecisions, termDecisions, modelDecision = decisions
        for k, v in groupDecisions.items():
            g = self.theGroups.get(k)
            if g is None:
                raise excep.biogemeError(
                    f'Unknown group of variables {k}. '
                    f'Existing groups: '
                    f'{list(self.theGroups.keys())}'
                )
            g.setDecisions(v)
        for k, v in termDecisions.items():
            u = self.theAlternatives.get(k)
            if u is None:
                raise excep.biogemeError(
                    f'Unkown alternative {u}. '
                    f'Existing alternatives: '
                    f'{list(self.utilities.keys())}'
                )
            if len(u.terms) != len(v):
                raise excep.biogemeError(
                    f'Utility of {k} has {len(u.terms)}'
                    f' terms and decisions are for '
                    f'{len(v)} terms'
                )
            for t, d in zip(u.terms, v):
                t.setDecisions(d)
        if modelDecision < 0 or modelDecision >= len(self.models):
            raise excep.biogemeError(
                f'Invalid model number: {modelDecision}.'
                f' Must be between 0 and '
                f'{len(self.models)-1}'
            )
        self.selectedModel = modelDecision

    def reset(self):
        """
        Deasactivate variables from the model, that can be deactivated.
        """
        for g in self.theGroups.values():
            g.activate(False)
        for seg in self.theSegmentations.values():
            decisions = seg.getDecisions()
            decisions = {x: False for x in decisions}
            seg.setDecisions(decisions)

    def generateSolution(self, nonlinearSpecs, segmentations, model):
        """Generate a solution for the VNS algorithm

        :param nonlinearSpecs: nonlinear specifications. It is a dictionary
            where

            - the keys correspond to groups of variables,
            - the values are tuple with two entries:

                - index in the list self.nonlinearSpecs corresponding to
                  the nonlinear spec, or None if linear,
                - a boolean that is True if the coefficient is generic,
                  False otherwise.

            Example::

                nl = {'Travel time': (0, False),
                      'Travel cost': (0, False),
                      'Headway': (0, False)}

        :type nonlinearSpecs: dict(str: tuple(int, bool))

        :param segmentations: dictionary where the keys are the name
            of the segmentations, and the values are lists of
            socio-economic characteristics that must be activated.

            Example::

                sg = {'Seg. cte': ['GA'],
                      'Seg. cost': ['class', 'who'],
                      'Seg. time': ['gender'],
                      'Seg. headway': ['class']}
        :type segmentations: dict(str: list(str))

        :param model: selected model
        :type model: str

        :return: the solution that has been generated
        :rtype: class solution

        :raise biogemeError: if a variable is set ot generic. Only groups of
            variables can be made generic.

        :raise biogemeError: if a group of variables is unknown

        :raise biogemeError: if an error occurs in setting the segmentation
            decisions

        :raise biogemeError: if the model is unknown.

        """
        self.reset()
        for k, (nl, generic) in nonlinearSpecs.items():
            if k in self.theGroups:
                self.theGroups[k].activate(True)
                self.theGroups[k].makeGeneric(generic)
                if nl is None:
                    self.theGroups[k].setLinear(True)
                else:
                    self.theGroups[k].setLinear(False)
                    self.theGroups[k].setSelection(nl)
            elif k in self.theVariables:
                self.theVariables[k].active = True
                self.theVariables[k].nonlinearSpecs = nl
                if generic:
                    raise excep.biogemeError(
                        f'Only groups of variables can '
                        f'be made generic, not {k}'
                    )
            else:
                raise excep.biogemeError(f'Unknown (group of) variable(s) {k}')

        for k, seg in segmentations.items():
            decisions = self.theSegmentations[k].getDecisions()
            decisions = {x: False for x in decisions}
            decisions = {x: True for x in seg}
            try:
                self.theSegmentations[k].setDecisions(decisions)
            except excep.biogemeError as e:
                error_msg = f'Error in segmentation of {k}: {e}'
                raise excep.biogemeError(error_msg) from e
        self.selectedModel = None
        for i, pair in enumerate(self.models):
            if pair[0] == model:
                self.selectedModel = i
        if self.selectedModel is None:
            raise excep.biogemeError(f'Unknown model {model}')
        return self.getSolution()

    def getSolution(self):
        """Generate an object of the class ``solution``

        :return: the solution that has been generated
        :rtype: class solution
        """
        s = solution()
        s.decisions = self.getDecisions()
        s.description = self.describeCurrentModel()
        s.objectives = self.archive.get(s)
        return s

    def setSolution(self, aSolution):
        """Import a solution defined as an object of class ``solution``.

        :param aSolution: solution to be imported.
        :type aSolution: class solution

        :raise biogemeError: if the object has the wrong type.
        """
        if not isinstance(aSolution, vns.solutionClass):
            raise excep.biogemeError(
                f'Wrong type: {type(aSolution)} '
                f'instead of vns.solutionClass'
            )

        self.setDecisions(aSolution.decisions)

    def clone(self):
        """Clone the model, in order to generate neighbors

        :return: a clone
        :rtype: specificationProblem
        """
        c = copy.deepcopy(self)
        return c

    def describeCurrentModel(self):
        """Generates a description of the current model

        :return: model description.
        :rtype: str
        """
        result = f'{self.models[self.selectedModel][0]}'
        for v in self.theAlternatives.values():
            title = f'Alternative {v.name} [{v.id}]\n'
            result += '-' * len(title)
            result += '\n'
            result += title
            result += '-' * len(title)
            result += '\n'
            for t in v.terms:
                result += t.describe()
                result += '\n'
        return result

    def describe(self, aSolution):
        """Generates (if necessary) and returns a short description of
            the solution

        :param aSolution: solution that must be described.
        :type aSolution: class solution

        :return: description of the solution
        :rtype: str
        """
        if aSolution.description is not None:
            return aSolution.description
        solution.description = self.describeCurrentModel()
        return solution.description

    def describeHtml(self):
        """Generates a description of the model in HTML format.

        :return: description of the model in HTML format.
        :rtype: str

        """
        result = (
            f'<p><strong>Model: {self.models[self.selectedModel][0]}'
            f'</strong></p>'
        )
        for v in self.theAlternatives.values():
            title = f'<p><strong>Alternative {v.name} [{v.id}]</strong></p>'
            result += title
            for t in v.terms:
                result += '<p>'
                result += html.escape(t.describe(), quote=True)
                result += '</p>'
                result += '\n'
        return result

    def isValid(self, aSolution):
        """Evaluate  the validity of the solution.

        :param aSolution: solution to be checked
        :type aSolution: class solution

        :return: valid, why where valid is True if the solution is
            valid, and False otherwise. ``why`` contains an explanation why it
            is invalid.
        :rtype: tuple(bool, str)
        """
        if aSolution.valid is not None:
            return aSolution.valid, aSolution.causeInvalidity

        try:
            self.setSolution(aSolution)
        except excep.biogemeError as e:
            logger.warning(f'Exception raised: {e}')
            aSolution.valid = False
            aSolution.causeInvalidity = e
            return False, e

        estimationResults = self.archive.get(aSolution)
        if estimationResults is None:
            estimationResults = self.evaluate(aSolution)
            if aSolution.valid is not None:
                return aSolution.valid, aSolution.causeInvalidity

        if estimationResults is None:
            aSolution.valid = False
            e = 'Failed estimation'
            aSolution.causeInvalidity = e
            return False, e

        if estimationResults.numberOfFreeParameters() == 0:
            aSolution.valid = False
            e = 'No parameter has been estimated'
            aSolution.causeInvalidity = e
            return False, e

        error_msg = ''
        ok = True
        for u in self.theAlternatives.values():
            for t in u.terms:
                tok, tmsg = t.isValid(u.name, estimationResults)
                if not tok:
                    ok = False
                    error_msg += f', {tmsg}'
        if ok:
            aSolution.valid = True
            return True, None

        aSolution.valid = False
        aSolution.causeInvalidity = error_msg
        return False, error_msg

    def evaluate(self, aSolution):
        """Evaluate the objectives functions of the solution and store them in
        the solution object.

        :param aSolution: solution to be evaluated
        :type aSolution: solutionClass

        :return: results of the estimation
        :rtype: class :class:`biogeme.results.bioResults`
        """
        estimationResults = self.archive.get(aSolution)
        if estimationResults is not None:
            return estimationResults

        self.setSolution(aSolution)

        b = self.getBiogemeModel()
        logger.detailed(
            f'Evaluate model with '
            f'{len(b.id_manager.free_betas.names)} parameters.'
        )
        if len(b.id_manager.free_betas.names) == 0:
            estimationResults = None
            self.archive[aSolution] = estimationResults
            aSolution.valid = False
            aSolution.causeInvalidity = 'Model with 0 parameters'
            return estimationResults

        if len(b.id_manager.free_betas.names) > self.maximumNumberOfParameters:
            estimationResults = None
            self.archive[aSolution] = estimationResults
            aSolution.valid = False
            aSolution.causeInvalidity = (
                f'More than {self.maximumNumberOfParameters}'
                f' parameters: {len(b.id_manager.free_betas.names)}'
            )
            return estimationResults

        algoParameters = {'proportionAnalyticalHessian': 0.0, 'maxiter': 200}
        try:
            logger.temporarySilence()
            estimationResults = b.quickEstimate(algoParameters=algoParameters)
            logger.resume()
            aSolution.objectives = [
                -estimationResults.data.logLike,
                estimationResults.numberOfFreeParameters(),
            ]

        except excep.biogemeError as e:
            logger.warning(f'Exception raised: {e}')
            estimationResults = None
            aSolution.valid = False
            aSolution.causeInvalidity = (
                f'Exception raised during estimation: {e}'
            )
        self.archive[aSolution] = estimationResults
        return estimationResults

    def checkAvailability(self):
        """Check the availability of each operator.

        :return: a dictionary with the availability status of each operator
        :rtype: dict(str: bool)

        """
        av = {k: v(audit=True) != 0 for k, v in self.operators.items()}
        return av

    def generateNeighbor(self, aSolution, neighborhoodSize):
        """
        Generates a neighbor of the solution.

        :param aSolution: solution to be modified
        :type aSolution: class solution

        :param neighborhoodSize: size of the neighborhood to be applied
        :type neighborhoodSize: int

        :return: a neighbor solution, and the number of changes that have been
                 actually applied.
        :rtype: tuple(class solution, int)
        """
        self.setSolution(aSolution)
        # Identify the operators available for the current specification
        self.operatorsManagement.available = self.checkAvailability()
        # Select one operator.
        self.lastOperator = self.operatorsManagement.selectOperator()
        logger.detailed(f'Apply operator {self.lastOperator}')
        changes = self.applyOperator(self.lastOperator, neighborhoodSize)
        return self.getSolution(), changes

    def neighborRejected(self, aSolution, aNeighbor):
        """Notify that a neighbor has been rejected by the
        algorithm. Used to update the statistics on the operators.

        :param aSolution: solution modified. Not used in this implementation.
        :type aSolution: class solution

        :param aNeighbor: neighbor
        :type aNeighbor: solutionClass

        :raise biogemeError: if no operator has been used yet.
        """
        if self.lastOperator is None:
            raise excep.biogemeError('No operator has been used yet.')
        self.operatorsManagement.decreaseScore(self.lastOperator)

    def neighborAccepted(self, aSolution, aNeighbor):
        """Notify that a neighbor has been accepted by the
        algorithm. Used to update the statistics on the operators.

        :param aSolution: solution modified. Not used in this implementation.
        :type aSolution: solutionClass

        :param aNeighbor: neighbor
        :type aNeighbor: solutionClass

        :raise biogemeError: if no operator has been used yet.

        """
        if self.lastOperator is None:
            raise excep.biogemeError('No operator has been used yet.')
        self.operatorsManagement.increaseScore(self.lastOperator)

    def applyOperator(self, name, size=1):
        """Apply an operator.

        :param name: name of the operator to apply
        :type name: str

        :param size: size of the neighborhood
        :type size: int

        :return: total number of changes actually made on the solution
        :rtype: int

        :raise biogemeError: if the name of the operator is unknown.
        """
        op = self.operators.get(name)
        if op is None:
            raise excep.biogemeError(f'Unknowns operator: {name}')
        return op(size)

    def changeSegmentation(self, size=1, audit=False):
        """
        Change the interaction, while keeping the number of them
        """
        totalChanges = 0
        segments = list(self.theSegmentations.keys())
        random.shuffle(segments)
        for s in segments:
            if not self.theSegmentations[s].isActive():
                changes = 0
            else:
                d = self.theSegmentations[s].getDecisions()
                active = [k for k, v in d.items() if v]
                inactive = [k for k, v in d.items() if not v]
                changes = min([size, len(active), len(inactive)])
                totalChanges += changes
                if not audit:
                    for i in range(changes):
                        newd = d
                        newd[active[i]] = False
                        newd[inactive[i]] = True
                        self.theSegmentations[s].setDecisions(newd)
                if totalChanges == size:
                    return totalChanges
        return totalChanges

    def increaseSegmentation(self, size=1, audit=False):
        """
        Add a level of segmentation.
        """
        candidates = []
        segments = list(self.theSegmentations.keys())
        for s in segments:
            if self.theSegmentations[s].isActive():
                d = self.theSegmentations[s].getDecisions()
                inactive = [k for k, v in d.items() if not v]
                candidates += [(s, k) for k in inactive]

        changes = min(size, len(candidates))
        if not audit:
            random.shuffle(candidates)
            for i in range(changes):
                s, k = candidates[i]
                d = self.theSegmentations[s].getDecisions()
                d[k] = True
                d = self.theSegmentations[s].setDecisions(d)

        return changes

    def decreaseSegmentation(self, size=1, audit=False):
        """
        Remove a level of segmentation.
        """

        candidates = []
        segments = list(self.theSegmentations.keys())
        for s in segments:
            if self.theSegmentations[s].isActive():
                d = self.theSegmentations[s].getDecisions()
                active = [k for k, v in d.items() if v]
                candidates += [(s, k) for k in active]

        changes = min(size, len(candidates))
        if not audit:
            random.shuffle(candidates)
            for i in range(changes):
                s, k = candidates[i]
                d = self.theSegmentations[s].getDecisions()
                d[k] = False
                d = self.theSegmentations[s].setDecisions(d)

        return changes

    def changeLinearity(self, size=1, audit=False):
        """
        Make linear if non linear, and the other way around.
        """
        groups = [g for g in self.theGroups.keys() if self.theGroups[g].active]
        random.shuffle(groups)
        changes = min([size, len(groups)])
        if not audit:
            for i in range(changes):
                self.theGroups[groups[i]].swapLinear()

        return changes

    def changeVariables(self, size=1, audit=False):
        """
        Activate groups of variables
        """
        groups = [
            g
            for g in self.theGroups.keys()
            if not self.theGroups[g].alwaysActive
        ]
        random.shuffle(groups)
        changes = min([size, len(groups)])
        if not audit:
            for i in range(changes):
                self.theGroups[groups[i]].swapActivate()
        return changes

    def changeGeneric(self, size=1, audit=False):
        """
        Change generic vs alternative specific status
        """
        groups = [
            g
            for g in self.theGroups.keys()
            if self.theGroups[g].active
            and not self.theGroups[g].genericForbiden
        ]
        random.shuffle(groups)
        changes = min([size, len(groups)])
        if not audit:
            for i in range(changes):
                self.theGroups[groups[i]].swapGeneric()
        return changes

    def changeNonlinearity(self, size=1, audit=False):
        """
        Change the nature of the nonlinear specification
        """
        groups = [
            g
            for g in self.theGroups.keys()
            if self.theGroups[g].active and not self.theGroups[g].linear
        ]
        random.shuffle(groups)
        changes = min([size, len(groups)])
        if not audit:
            for i in range(changes):
                # We need to select randomly a specification different
                # from the current one
                nbrOfValues = len(self.theGroups[groups[i]].nonlinearSpecs)
                values = set(range(nbrOfValues))
                values.remove(self.theGroups[groups[i]].selection)
                sel = random.choice(list(values))
                self.theGroups[groups[i]].setSelection(sel)

        return changes

    def changeModel(self, size=1, audit=False):
        """
        Select randomly another model from the list

        :param size: not used here. Must be there for compliance with the
            call of operators.
        :type size: int

        :param audit: if True, returns the number of changes without
            actually implementing them.
        :type audit: bool

        :return: 0 if no other model could be found. 1 otherwise.
        :rtype: int
        """
        candidates = set(range(len(self.models)))
        candidates.remove(self.selectedModel)
        if not candidates:
            return 0
        if not audit:
            self.selectedModel = random.choice(list(candidates))
        return 1


class solution(vns.solutionClass):
    """
    Class representing one solution, that is, one model specification.
    """

    def __init__(self):
        super().__init__()

        self.objectivesNames = ['Neg. log likelihood', '#parameters']
        """ Names of the objective functions
        """

        self.objectives = None  #: values of the objectives

        self.valid = None  #: True if the solution is valid

        self.causeInvalidity = None
        """If the solution is invalid, contains the cause of the invalidity
        """

        self.decisions = None
        """
        The decisions consist of:

        - a dict of decisions for each group of variables.
        - for each utility, a list of decisions for each term.
        - the selected model

        Type: tuple(dict(str: int), dict(str: list(dict(str: bool))), int)
        """

        self.description = None  #: description of the solution

    def __repr__(self):
        return str(self.decisions)

    def __str__(self):
        if self.description is None:
            raise excep.biogemeError(
                'Description of the solution not available'
            )
        res = self.description
        if self.objectivesNames is None:
            raise excep.biogemeError(
                'The attribute objectivesNames is not defined'
            )
        if self.objectives is None:
            raise excep.biogemeError('The attribute objectives is not defined')
        for t, r in zip(self.objectivesNames, self.objectives):
            res += f'\n{t}: {r}'
        return res
