//-*-c++-*------------------------------------------------------------
//
// File name : patModelSpec.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Nov  7 14:12:56 2000
//
//--------------------------------------------------------------------

#ifndef patModelSpec_h
#define patModelSpec_h

#include <vector>
#include <list>
#include <map>
#include <set>
#include "patFormatRealNumbers.h"
#include "patThreeStrings.h"
#include "patStlVectorIterator.h"
#include "patError.h"
#include "patVariables.h"
#include "patIterator.h"
#include "patBetaLikeParameter.h"
#include "patAlternative.h"
#include "patNlNestDefinition.h"
#include "patCnlAlphaParameter.h"
#include "patEstimationResult.h"
#include "patNetworkGevLinkParameter.h"
#include "patLinearConstraint.h" 
#include "patNonLinearConstraint.h"
#include "patRandomParameter.h"
#include "patDiscreteParameter.h"
#include "patOneZhengFosgerau.h"
#include "patSequenceIterator.h"

#include "patLoop.h"

class patGEV ;
class patNL ;
class patArithNode ;
class patArithVariable ;
class patParamManagement;
class patNetworkGevModel ;
class patNetworkGevNode ;
class patUtility ;
class patArithRandom ;
class patSecondDerivatives ;
class trHessian ;
class patPythonResults ;
class patSampleEnuGetIndices ;
/**
   @doc This class is designed to read the model specifications from a file and
   make them available to all other objects. A single instance of the class is
   available at any time (singleton pattern).
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Nov  7 14:12:56 2000)
 */

class patModelSpec {

  friend class patBisonSingletonFactory ;
  friend class patSpecParser ;
  friend class patBisonSpec ;
  friend class PyModel ;

public:
  /**
     Typically, this is set to $ROOT as in the parser, but it does not have to.
   */
  const patString rootNodeName ;

  /**
   */
  struct patConstantProduct {
    /**
     */
    patConstantProduct(const patString& p1, 
		       const patString& p2,
		       patReal c) : param1(p1), param2(p2), cte(c) {}
    /**
     */
    patString param1 ;
    /**
     */
    patString param2 ;
    /**
     */
    patReal cte ;
  };

  /**
   */
  struct patConstantProductIndex {
    /**
     */
    unsigned long index1 ;
    /**
     */
    unsigned long index2 ;
    /**
     */
    patReal cte ;
  };

public:


  /**
   */
  virtual ~patModelSpec() ;

  /**
     Erase everything and create a new object
   */
  void reset() ;

  /**
   */
  typedef enum { patOLtype,
                 patBPtype,
		 patMNLtype, 
		 patNLtype,
		 patCNLtype,
		 patNetworkGEVtype
  } patModelType ;


  /**
     @param reset: if true, the single instance is destroyed and constructed again
     @return pointer to the single instancle of the class
   */
  static patModelSpec* the() ;

  /**
     @param fileName name of the specification file
     @param err ref. of the pointer to the error object.
   */
  virtual void readFile(const patString& fileName, 
			patError*& err) ;

  /**
     Read the first line of the the data file
     @param err ref. of the pointer to the error object.
   */
  virtual void readDataHeader(patError*& err) ;

  /**
   */
  virtual void setDataHeader(vector<patString>& header) ;

  /**
   */
  unsigned long getNumberOfDraws() ;

  /**
     get the temporary number of draws controlled by the algorithm
   */
  unsigned long getAlgoNumberOfDraws() ;

  /**
     set the temporary number of draws controlled by the algorithm
   */
  void setAlgoNumberOfDraws(unsigned long d) ;

  /**
   */
  unsigned long getNbrDrawAttributesPerObservation() ;

  /**
     @param model name
   */
  void setModelName(const patString& name) ;

  /**
     @return model name
   */
  patString getModelName() ;

  /**
     Check if a given keyword is defined in the header of the data file
   */
  virtual patBoolean isHeader(patString name) const ;

  /**
     @return patTRUE if the model is binary probit
   */
  virtual patBoolean isBP() const ;
  /**
     @return patTRUE if the model is ordinal logit
   */
  virtual patBoolean isOL() const ;
  /**
     @return patTRUE if the model is multinomial 
   */
  virtual patBoolean isMNL() const ;
  /**
     @return patTRUE if the model is nested
   */
  virtual patBoolean isNL() const;
  /**
     @return patTRUE if the model is cross-nested
   */
  virtual patBoolean isCNL() const ;
  /**
     @return patTRUE if the model is Network GEV
   */
  virtual patBoolean isNetworkGEV() const ;

  /**
     @return patTRUE if the model is a GEV model
   */
  virtual patBoolean isGEV() const ;

  /**
     @return patTRUE if the model contains random coefficients
   */
  virtual patBoolean isMixedLogit() const ;

  /**
     Assign alternatives to nests for the nested logit model
     @param nestedModel ponter to the nested logit
     @param err ref. of the pointer to the error object.
   */
  virtual void assignNLNests(patNL* nestedModel,
			     patError*& err) const ;

  /**
   */
  virtual unsigned long getNbrAlternatives() const ;
  /**
   */
  virtual unsigned long getNbrNests() const ;
  /**
   */
  virtual unsigned long getNbrAttributes() const ;
  /**
   */
  virtual unsigned long getNbrUsedAttributes() const ;
  /**
   */
  virtual unsigned long getNbrRandomParameters() const ;
  /**
   */
  virtual unsigned long getNbrDataPerRow(unsigned short fileId) const ; 
  /**
   */
  virtual unsigned long getNbrHeaders(unsigned short nh) const ; 

  /**
   */
  virtual vector<patString>* getHeaders(unsigned short fileId) ;
  /**
     @return number of unknown parameters in the utility functions. It does
     not take into account the parameters from the Box-Tukey transform. 
  */
  virtual unsigned long getNbrOrigBeta() const ;


  /**
     @return Number of factors  for the Error Components Logit Kernel 
   */
  unsigned long getNbrFactors() const;

  /**
     @return number of non zero entries in the factor loading matrix for LK
   */
  unsigned long getNonZeroEntriesFactorLoadingMatrix() ;

  /**
   */
  unsigned long getNbrModelParameters() const ;
  /**
   */
  unsigned long getNbrScaleParameters() const ;

  /**
     @return total number of unknown parameters in the utility functions, including parameters from the Box-Tukey transform. 
  */
  virtual unsigned long getNbrTotalBeta() const ;

  /**
   */
  virtual unsigned long getNbrNonFixedBetas() const ;

  /**
   */
  virtual unsigned long getNbrNonFixedParameters() const ;

  /**
     @return description of the linear-in-parameter utility function
     @param user id of an alternative.
     @param err ref. of the pointer to the error object.
  */
  virtual patUtilFunction* getUtil(unsigned long altId,
				   patError*& err) ;

  /**
     The formula is exactly as in the input file, while the utilities
     are modified into linear form for efficiency sake
     @return description of the formula of the 
     linear-in-parameter utility function 
     @param  user id of an alternative.  
     @param err ref. of the pointer to the
     error object.
  */
  virtual patUtilFunction* getUtilFormula(unsigned long altId,
					  patError*& err) ;

  /**
     @return utility function to be used in the model evaluation
  */
  patUtility* getFullUtilityFunction(patError*& err) ;

  /**
     @return patBadId if the beta parameter is fixed. If betaName is not known, an error is triggered.
   */
  virtual unsigned long getBetaIndex(const patString& betaName, 
				    patBoolean* exists) const ;

  /**
     @return user ID of an alternative. User IDs are read from the file, and
     consistent with the choice number
     @param altInternalId internal ID of an alternative
     @param err ref. of the pointer to the error object.
  */
  virtual unsigned long getAltId(unsigned long altInternalId,
			patError*& err) ;
  
  /**
     @return index of the attribute
     @param attrName name of the attribute
   */
  unsigned long getAttributeId(const patString attrName) ;

  /**
     @return index of the used attribute (that is an attribute
     explicitly used in a utility function
     @param attrName name of the  attribute
   */
  unsigned long getUsedAttributeId(const patString attrName, patBoolean* found) ;
  /**
     @return index of a random draw attribute
     @param attrName name of the  attribute
   */
  unsigned long getRandomAttributeId(const patString attrName, patBoolean* found) ;
  
  /**
     @return name of the attribute
     @param attrId index of the attribute
  */
  patString getAttributeName(unsigned long attrId) ;
  

  /**
     @return patTRUE if thre alternative exists 
     @param altId user ID of an alternative
  */
  patBoolean doesAltExists(unsigned long altUserId) ;
  /**
     @return name of an alternative, given its user ID, or "BIOGEME__UnknownAlt" if unknown
     @param altId user ID of an alternative
  */
  virtual patString getAltName(unsigned long altId,
				  patError*& err) ;
  

  /**
     @return internal ID of the alternative
     @param name 
   */
  virtual unsigned long getAltInternalId(const patString& name) ;
  
  /**
     @return user ID of the alternative
     @param name 
   */
  virtual unsigned long getAltUserId(const patString& name) ;
  
  /**
     @return name of the availability description, given its user ID.
     @param altId user ID of an alternative
     @param err ref. of the pointer to the error object.
  */
  virtual patString getAvailName(unsigned long altId,
				  patError*& err) ;
  
  /**
     @return internal ID of an alternative.  Internal IDs are
     designed to be consecutive from 0 to the nbr of alt - 1
     consistent with the choice number
     @param altId User ID of an alternative
     @param err ref. of the pointer to the error object.
  */

  virtual unsigned long getAltInternalId(unsigned long altId,
			patError*& err) ;
  

  /**
     @return column number in the data file of a specific header
     @param headerName
     @param err ref. of the pointer to the error object.
   */
  virtual unsigned long getHeaderRank(unsigned short fileId, 
				      const patString& headerName,
				      patError*& err) ;


  /**
     @return headerName
     @param rank column number in the data file of a specific header
     @param err ref. of the pointer to the error object.
   */
  virtual patString getHeaderName(unsigned short fileId,
				  unsigned long rank,
				  patError*& err) ;

  /**
     reset default values for the headers
   */
  void resetHeadersValues() ;

  /**
     @return pointer to the arithmetic expression computing the choice
     @param err ref. of the pointer to the error object.
   */
  virtual patArithNode* getChoiceExpr(patError*& err) ;
  /**
     @return pointer to the arithmetic expression computing the
     boolean checking if it is the last aggregate observation
      
     @param err ref. of the pointer to the error object.
   */

  virtual patArithNode* getAggLastExpr(patError*& err) ;
  /**
     @return pointer to the arithmetic expression computing the aggregate 
      weight
     @param err ref. of the pointer to the error object.
   */
  virtual patArithNode* getAggWeightExpr(patError*& err) ;
  /**
     @return pointer to the arithmetic expression computing the weight
     @param err ref. of the pointer to the error object.
   */
  virtual patArithNode* getWeightExpr(patError*& err) ;

  /**
     @return pointer to the arithmetic expression computing the
     individual id for panel data
     @param err ref. of the pointer to     the error object.
   */
  virtual patArithNode* getPanelExpr(patError*& err) ;

  /**
     @return pointer  covariance parameter, or NULL if not found
     @param rv1 first random variable
     @param rv2 second random variable
   */
  patBetaLikeParameter* getCovarianceParameter(const patString& rv1,
					      const patString& rv2) ;

  /**
     @return patTRUE if the sample must be weighted
  */
  patBoolean isSampleWeighted() ;

  /**
     @return pointer to the arithmetic expression computing the group
     @param err ref. of the pointer to the error object.
   */
  virtual patArithNode* getGroupExpr(patError*& err) ;

  /**
     @return pointer to the arithmetic expression computing the exclusion condition
     @param err ref. of the pointer to the error object.
   */
  virtual patArithNode* getExcludeExpr(patError*& err) ;
  
  /**
     @return pointer to the arithmetic expression computing the nonlinear part of a utility function
     @param altId identifier of the alternative
     @param err ref. of the pointer to the error object.
   */
  patArithNode* getNonLinearUtilExpr(unsigned long altId,
				     patError*& err) ;
  

  /**
     @return pointer to the arithmetic expression computing the derivative of the nonlinear part of a utility function
     @param altId identifier of the alternative
     @param param derivative with respect to this param
     @param err ref. of the pointer to the error object.
   */
  patArithNode* getDerivative(unsigned long userAltId,
			      patString param,
			      patError*& err) ;


  /**
     @return TRUE is the analytical derivatives of the nonlinear utilities are available from the user
   */
  patBoolean utilDerivativesAvailableFromUser() ;

  /**
     @return pointer to the arithmetic expression computing the derivative of the nonlinear part of a utility function
     @param altId identifier of the alternative
     @param betaIndex derivative with respect  param id
     @param err ref. of the pointer to the error object.
   */
  patArithNode* getDerivative(unsigned long internalAltId,
			      unsigned long betaId,
			      patError*& err) ;
  

  /**
     This function returns the arithmetic expression of a variable. If
     the variable has not been explicitly defined, it searches the
     headers of the data file, and then in the headers of the draws
     file. If the variable is neither in the expressions neither in
     the headers, an error is generated and a NULL pointer is
     returned.  @return pointer to the arithmetic expression computing
     a variable @param name name of the variable @param err ref. of
     the pointer to the error object.
   */
  virtual patArithNode* getVariableExpr(const patString& name,
				patError*& err) ;


  /**
     This function outputs to a stream the variable definitions.
   */
  virtual void printVarExpressions(ostream& str,patBoolean forPython,patError*& err) ;

  /**
     @return pointer to the arithmetic expression computing the  availability 
                    of an alternative
     @param fileId id of the data file used to compute the expression
     @param altid user ID of the alternative 
     @param err ref. of the pointer to the error object.
   */
  virtual patArithNode*  getAvailExpr(unsigned long altId,
				      patError*& err) ;

  /**
     @doc Debugging procedure printing all known expressions on cout
   */
  virtual void printExpressions(patError*& err) ;

  /**
     @doc return a parameter, without a priori knowing its type
   */
  virtual patBetaLikeParameter getParameter(const patString& name,
					    patBoolean* found) const ;
  /**
   */
  virtual patBetaLikeParameter getBeta(const patString& name,
				       patBoolean* betaFound) const ;

  /**
     @param user id of the scale parameter
   */
  virtual patBetaLikeParameter getScale(unsigned long userId,
					patError*& err) const ;
  /**
     @param internal id of the scale parameter
   */
  virtual patBetaLikeParameter getScaleFromInternalId(unsigned long id,
						  patError*& err) const ;
  /**
   */
  virtual patBetaLikeParameter getMu(patError*& err) const ;

  /**
     return NULL if the model is not a Generalized Extreme Value 
   */
  patBetaLikeParameter* getMGEVParameter() ;

  /**
   */
  patBoolean isMuFixed() const ;

  /**
   */
  virtual patBetaLikeParameter getNlNest(const patString& name,
					 patBoolean* found) const ;
  
  /**
   */
  virtual patBetaLikeParameter getCnlNest(const patString& name,
					  patBoolean* found) const  ;

  /**
   */
  virtual patBetaLikeParameter getCnlAlpha(const patString& nestName,
					   const patString& altName,
					   patBoolean* found) const ;
  /**
   */
  virtual patBetaLikeParameter getNetworkNode(const patString& name,
					      patBoolean* found) const ;

  /**
   */
  virtual patBetaLikeParameter getNetworkLink(const patString& name,
					      patBoolean* found) const ;

  /**
   */
  patString modelTypeName() const ;

/**
 */
  friend ostream& operator<<(ostream &str, const patModelSpec& x) ;

  /**
     From the name of a CNL alpha parameter, this routine returns the names of
     (i) the alternative and (ii) the nest
  */
  virtual pair<patString,patString> getCnlAlphaAltNest(const patString& name,
						       patBoolean* found) const ;

  /** From the name of an alternative and a nest, build the name of the
      corresponding alpha parameter.  */

  patString buildAlphaName(const patString& altName, const patString& nestName) const ;
  
  /** From the name of two nodes, build the name of the corresponding link
      parameter in the network GEV model.  */
  patString buildLinkName(const patString& aNode, const patString& bNode) const ;
  /**
     Build the name of a random parameter from the name of its mean and the name of its stddev
   */
  patString buildRandomName(const patString& meanName, const patString& stdDevName) const ;

  /**
     Build the name of a covariance parameter from the name of two random variables 
   */
  patString buildCovarName(const patString& rv1, const patString& rv2) const ;

  /**
   */
  patString getDiscreteParamName(unsigned long i, patError*& err) ;

  /**
   */
  unsigned long getBetaId(const patString& name, patError*& err) ;
  /**
   */
  void computeIndex(patError*& err) ;

  /**
     This set coefficient values coming from the algorithms. This function
     must be called before the likelihood can be computed.
  */
  void setEstimatedCoefficients(const patVariables* x,patError*& err) ;

  /**
  */
  patVariables getEstimatedCoefficients(patError*& err) ;

  /**
     Id is the index of the CNL parameters in the model parameters vector
   */
  unsigned long getIdCnlNestCoef(unsigned long nest) ;

  /**
   */
  unsigned long getIdCnlAlpha(unsigned long nest,
				  unsigned long alt) const ;

  /**
   */
  patBoolean isCnlParamNestCoef(unsigned long index) ;

  /**
   */
  patBoolean isCnlParamAlpha(unsigned long index) ;

  /**
   */
  pair<unsigned long,unsigned long> getNestAltFromParamIndex(unsigned long index) ;

  /**
   */
  patVariables* getPtrBetaParameters() ;
  /**
   */
  patReal* getPtrMu() ;
  /**
   */
  patVariables* getPtrModelParameters() ;
  /**
   */
  patVariables* getPtrScaleParameters() ;

  /**
     Iterates on the names of attributes
   */
  patIterator<patString>* createAttributeNamesIterator() ;

  /**
     Iterates on the names of used attributes
   */
  patIterator<pair<patString,unsigned long> >* createUsedAttributeNamesIterator() ;

  /**
     All parameters potentially estimable
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createAllParametersIterator() ;
  

  /**
     WARNING: the caller is responsible to release the allocated memory
  */
  patIterator<patBetaLikeParameter>* createAllBetaIterator() ;
  /**
     All model means GEV specific parameters, like the nest coefficients and
     the alphas for the CNL
     WARNING: the caller is responsible to release the
     allocated memory 
  */
  patIterator<patBetaLikeParameter>* createAllModelIterator() ;
  /**
     Creates an iterator on the pair on nest parameters ehich must be
     constrainted to be equal
     WARNING: the caller is responsible to release the
     allocated memory 
  */
  patIterator<pair<unsigned long, unsigned long> >* createConstraintNestIterator(patError*& err) ; 

  /**
     Creates an iterator on constraints of type param1*param2=cte
     WARNING: the caller is responsible to release the
     allocated memory 
  */
  patIterator<patConstantProductIndex>* createConstantProductIterator(patError*& err) ;

  /**
     WARNING: the caller is responsible to release the allocated memory
  */
  patIterator<patBetaLikeParameter>* createBetaIterator()   ;
  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createScaleIterator() ;
  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createNlNestIterator() ;
  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createCnlNestIterator() ;
  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createCnlAlphaIterator() ;

  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patCnlAlphaParameter>* createFullCnlAlphaIterator() ;

  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createNetworkNestIterator() ;
  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createNetworkAlphaIterator() ;
  
  /**
     WARNING: the caller is responsible to release the allocated memory
   */
  patIterator<patBetaLikeParameter>* createScalesIterator() ;


  /**
   */
  void writeReport(patString fileName, patError*& err) ;

  /**
   */
  void writeALogit(patString fileName, patError*& err) ;

  /**
   */
  void writeHtml(patString fileName, patError*& err) ;

  /**
   */
  void writePythonResults(patPythonResults* pythonRes, patError*& err) ;

  /**
   */
  void writePythonSpecFile(patString fileName, patError*& err) ;

  /**
   */
  void writeLatex(patString fileName, patError*& err) ;

  /**
   */
  void writeSpecFile(patString fileName, patError*& err) ;

  /**
   */
  void saveBackup() ;

  /**
   */
  void writeGnuplotFile(patString fileName, 
			patError*& err) ;


  /**
   */
  void setEstimationResults(const patEstimationResult& res) ;

  /**
     Given derivatives with regards to each parameter, build the gradient of
     the log-likelihood function.
     It also build BHHH if requested, and the true Hessian if available. 
     All values are accumulated.
  */

  patVariables* gatherGradient(patVariables* grad,
			       patBoolean computeBHHH,
			       vector<patVariables>* bhhh,
			       trHessian* trueHessian,
			       patVariables* betaDerivatives,
			       patReal* scaleDerivative,
			       unsigned long scaleIndex,
			       patVariables* paramDerivative,
			       patReal* muDerivative,
			       patSecondDerivatives* secondDeriv,
			       patError*& err) ;
  
  /**
     Not for biogeme...
   */
  patVariables* gatherGianlucaGradient(patVariables* grad,
				       patVariables* betaDerivatives,
				       patError*& err) ;
  
  /**
   */
  patVariables getLowerBounds(patError*& err) const ;
    
  /**
   */
  patVariables getUpperBounds(patError*& err) const ;
  /**
     Given a nest, this methods creates an iterator on the alternatives such
     that the CNL Alpha coefficient (nest,alt) is non zero.
  */
  patIterator<unsigned long>* 
  getAltIteratorForNonZeroAlphas(unsigned long nest) ;
  /**
     Given an alternative, this methods creates an iterator on the nests such
     that the CNL Alpha coefficient (nest,alt) is non zero.
   */
  patIterator<unsigned long>* 
  getNestIteratorForNonZeroAlphas(unsigned long alt) ;

  /**
   */
  patIterator<patDiscreteParameter*>* getDiscreteParametersIterator() ;

  /**
     @return patBadId if the nodeName is not known
   */
  unsigned long getNetworkGevNodeId(const patString&  nodeName) ;

  /**
     @return  pointer to the Network GEV model.
   */
  patGEV* getNetworkGevModel() ;

  /**
   */
  unsigned long nbrSimulatedChoiceInSampleEnumeration() ;

  /**
  */
  void copyParametersValue(patError*& err) ;

  /**
   */
  patListProblemLinearConstraint getLinearEqualityConstraints(patError*& err) ;

  /**
   */
  patListProblemLinearConstraint getLinearInequalityConstraints(patError*& err) ;
  
  /**
   */
  patListNonLinearConstraints* getNonLinearEqualityConstraints(patError*& err) ;

  /**
   */
  patListNonLinearConstraints* getNonLinearInequalityConstraints(patError*& err) ;

  /**
   */ 
  patString printEqConstraint(patProblemLinearConstraint aConstraint) ;

  /**
   */ 
  patString printIneqConstraint(patProblemLinearConstraint aConstraint) ;


  /**
     Define the beta as variable to evaluate the expression of nonlinear utilities
   */
  void setVariablesBetaWithinExpression(patArithNode* expression) ;

  /**
     @param n number of inbdividuals
     Generate random numbers
   */

  void generateRandomNumbers(unsigned long n, patError*& err) ;

  /** Read list of summary parameters
   */
  void readSummaryParameters(patError*& err) ;

  /**
     Write one line in the summary file containing:
     0) The date and time
     1) The model name 
     2) The report file name
     3) The final log likelihood
     4) The sample size (soon...)
     5) The estimated value and t-test of parameters in the summary.lis file
     6) The exclusion condition
   */
  void writeSummary(patString fileName, patError*& err) ;

  /**
     This function adds an attribute for the constant one, if it does not exists
   */
  void createExpressionForOne() ;

  /**
   */
  patBetaLikeParameter getParameterFromIndex(unsigned long index,
					     patBoolean* found);

  /**
   */
  patBetaLikeParameter getParameterFromId(unsigned long id,
					  patBoolean* found);

  /**
     Identifies scales per beta parameter
   */
  void identifyScales(patVariables* attributeScales,patError*& err) ;


  /**
   */
  void scaleBetaParameters() ;

  /**
   */
  void unscaleBetaParameters() ;

  /**
   */
  void unscaleMatrix(patMyMatrix* matrix, patError*& err) ;

  /**
   */
  patVariables* getAttributesScale() ;

  /**
   */
  patDistribType getDistributionOfDraw(unsigned long i, patError*& err) ; 

  /**
     Check if a specific draw corresponds to an individual specific
     variable in a panel data context
  */
  patBoolean isDrawPanel(unsigned long i, patError*& err) ;

  /**
     Get the mass at zero for  a specific draw 
   */
  patReal getMassAtZero(unsigned long i, patError*& err) ;
  
  /**
   */
  patBoolean isPanelData() const ;

  /**
   */
  patBoolean isGianluca() const ;

  /**
   */
  patBoolean isAggregateObserved() const ;

  /**
     Check if all betas are used in utility functions
   */
  patBoolean checkAllBetaUsed(patError*& err) ;

  /**
   */
  void setDiscreteParameterValue(patString param,
				 patBetaLikeParameter* beta,
				 patError*& err) ;

  /**
   */
  patBoolean containsDiscreteParameters() const ;

  /**
   */
  void generateFileForDenis(patString fileName) ;


  /**
   */
  list<patString> getMultiLineModelDescription() ;

  /**
   */
  vector<patString> getHeaders() ;

  /**
   */
  patBoolean correctForSelectionBias() ;

  /**
   */
  unsigned long getIdOfSelectionBiasParameter(unsigned long altIntId,
					      patError*& err) ;


  /**
   */
  unsigned long getIdSnpAttribute(patBoolean* found) ;

  /**
   */
  patBoolean applySnpTransform() ;

  /**
   */
  unsigned short numberOfSnpTerms() ;

  /**
   */
  vector<unsigned long> getIdOfSnpBetaParameters() ;

  /**
     @param i is a number between 0 and numberOfSnpTerms-1
     @return returns the order of the polynomial term. 
   */
  unsigned short orderOfSnpTerm(unsigned short i, patError*& err) ;


  /**
     @param name of a random parameter
     @return TRUE is it appears in the panel data section
   */
  patBoolean isIndividualSpecific(const patString& rndParam) ;

  /**
     @return patTRUE is group scales are estimated
   */
  patBoolean estimateGroupScales() ;
  /**
     @return patTRUE is group scales are all fixed to one
   */
  patBoolean groupScalesAreAllOne() ;
  
  /**
     @return patTRUE if the model is a simple MNL, where the utility
     are linear-in-parameters. No scale, no mu, no panel, etc.
   */
  patBoolean isSimpleMnlModel() ;

  /**
     Return the number of Zheng-Fosgerau tests requested by the
     user. If probabilities is not NULL, the number of such test which
     are probabilities are reported in this variable. The first call
     to this function also assigns IDs to the expression to identify
     them in the output of the sample enumeration.
   */

  unsigned short numberOfZhengFosgerau(unsigned short* nbrOfProba) ;

  
  /**
   */ 
  vector<patOneZhengFosgerau>* getZhengFosgerauVariables() ;

  /**
   */
  void computeZhengFosgerau(patPythonReal** arrayResult,
			    unsigned long resRow,
			    unsigned long resCol,
			    patSampleEnuGetIndices* ei,
			    patError*& err) ;
  /**
   */
  patBoolean includeUtilitiesInSimulation() const ;

protected:

  /**
     Check if an expression can be evaluated using the headers of a given file
  */

  patBoolean checkExpressionInFile(patArithNode* expression,
				   unsigned short fileId,
				   patError*& err) ;

  /**
   This routine computes the variance-covariance matrix of the
   estimated normally distributed random parameters
   */
  void computeVarCovarOfRandomParameters(patError*& err) ;

  /**
   */
  void setNumberOfDraws(unsigned long d) ;

  /**
   */
  virtual void setChoice(patArithNode* choice) ;

  /**
   */
  virtual void setAggregateLast(patArithNode* aggLast) ;

  /**
   */
  virtual void setAggregateWeight(patArithNode* aggWeight) ;

  /**
   */
  virtual void setPanel(patArithNode* panel) ;

  /**
   */
  virtual void setPanelVariables(list<patString>* panelVar) ;

  /**
   */
  virtual void setModelDescription(list<patString>* md) ;

  /**
   */
  virtual void setWeight(patArithNode* weight) ;

  /**
   */
  virtual void addBeta(const patString& name,
	       patReal defaultValue,
	       patReal lowerBound,
	       patReal upperBound,
	       patBoolean isFixed) ;


  /**
   */
  void addIIATest(patString name, const list<long>* listAltId) ;

  /**
   */
  void addProbaStandardError(patString b1, patString b2, patReal value, patError*& err) ;

  /**
   */
  void addZhengFosgerau(patOneZhengFosgerau zf, patError*& err) ;

  /**
   */
  void setGeneralizedExtremeValueParameter(patString name) ;

  /**
   */
  void setRegressionObservation(const patString& name) ;

  /**
   */
  void setStartingTime(const patString& name) ;

  /**
   */
  void addAcqRegressionModel(const patUtilFunction* f) ;

  /**
   */
  void setAcqSigma(const patString& a) ;

  /**
   */
  void setValSigma(const patString& a) ;

  /**
   */
  void addValRegressionModel(const patUtilFunction* f) ;

  /**
   */
  void setDurationParameter(const patString& p1) ;

  /**
   */
  void addLatexName(const patString& coeffName,
		    const patString& latexName) ;

  /**
     Add a ratio of coefficients to be listed in the results
   */
  virtual void addRatio(const patString& num,
			const patString& denom,
			const patString& name) ;

  /**
   */
  virtual void setMu(patReal defaultValue,
	     patReal lowerBound,
	     patReal upperBound,
	     patBoolean isFixed) ;

  /**
     @param number of simulated choices to include in sample enumeration
  */
  virtual void setSampleEnumeration(long s) ;

  /**
     @param id user id of the alternative
     @param name name of the alternative
     @param availHeader name of the expression defining the availability 
            of the alternative
     @param function pointer to the utility function
   */
  virtual void addUtil(unsigned long id, 
	       const patString& name, 
	       const patString& availHeader,
		       const patUtilFunction* function, patError*& err) ;

  /**
     @param id user id of the alternative
     @param util expression for the nonlinear term of the utility
   */
  virtual void addNonLinearUtility(unsigned long id,
				   patArithNode* util) ;

  /**
     @param id user id of the alternative
     @param param name of the param by which the utility is derived
     @param util expression for the derivative o the nonlinear term of the utility
   */
  virtual void addDerivative(unsigned long id,
			     patString param,
			     patArithNode* util) ;

  /**
   */
  virtual void setGroup(patArithNode* group) ;

  /**
   */
  virtual void setExclude(patArithNode* group) ;

  /**
     When no group is defined by the user, a default group is defined here
   */
  void setDefaultGroup() ;

  /**
   */
  virtual void addScale(long groupId,
		patReal defaultValue,
	       patReal lowerBound,
	       patReal upperBound,
		patBoolean isFixed) ;

  /**
   */
  virtual void setModelType(patModelType mt) ;


  /**
   */
  virtual void addNest(const patString& name,
	       patReal defaultValue,
	       patReal lowerBound,
	       patReal upperBound,
	       patBoolean isFixed,
	       const list<long>* listAltId) ;

  /**
   */
  virtual void addCNLNest(const patString& name,
		  patReal defaultValue,
		  patReal lowerBound,
		  patReal upperBound,
		  patBoolean isFixed) ;
  
  /**
   */
  virtual void addCNLAlpha(const patString& altName,
		   const patString& nestName,
		   patReal defaultValue,
		   patReal lowerBound,
		   patReal upperBound,
		   patBoolean isFixed) ;

  /**
   */
  virtual void addCovarParam(const patString& param1,
			     const patString& param2,
			     patReal defaultValue,
			     patReal lowerBound,
			     patReal upperBound,
			     patBoolean isFixed) ;

  /**
   */
  void addNetworkGevNode(const patString& name,
			 patReal defaultValue,
			 patReal lowerBound,
			 patReal upperBound,
			 patBoolean isFixed) ;
		   
  /**
   */
  void addNetworkGevLink(const patString& aNodeName,
			 const patString& bNodeName,
			 patReal defaultValue,
			 patReal lowerBound,
			 patReal upperBound,
			 patBoolean isFixed) ;
		   
  /**
   */
  virtual void addAttribute(const patString& attrName) ;

  /**
   */
  virtual void addExpression(const patString& name, 
			     patArithNode* expr,
			     patError*& err) ; 

  /**
     Add a list of expressions defined within a loop
   */
  virtual void addExpressionLoop(const patString& name, 
				 patArithNode* expr,
				 patLoop* theLoop,
				 patError*& err) ; 

  /**
     Add a consstraint imposing that two nest coefficients are equal
   */
  virtual void addConstraintNest(const patString& firstNest,
				 const patString& secondNest) ;

  /**
     Add a constraint imposing that the product of two parameters is equal to a constant
   */
  virtual void addConstantProduct(const patString& firstParam,
				  const patString& secondParam,
				  patReal value) ;


  /**
     Add a discrete parameter
   */
  void addDiscreteParameter(const patString& paramName,
			    const vector<patThreeStrings >& listOfTerms,
			    patError*& err) ;

  /**
   */
  void addSelectionBiasParameter(const unsigned long alt,
				 const patString& paramName) ;

  /**
     Set the number of columns to be read in the data file
   */
  virtual void setColumns(unsigned long i) ;

  /**
     Set the list of linear constraint
   */
  void setListLinearConstraints(patListLinearConstraint* ptr) ;

  /**
     Set the list of nonlinear equality constraints
   */
  void setListNonLinearEqualityConstraints(patListNonLinearConstraints* ptr) ;

  /**
     Set the list of nonlinear equality constraints
   */
  void setListNonLinearInequalityConstraints(patListNonLinearConstraints* ptr) ;

  /**
   */
  void setGnuplot(patString x, patReal l, patReal u) ;

  /**
     @doc Add an expression for a random parameter @return NULL if p
     is a new parameter to be inserted, or a pointer to the existing
     parameter if it already exists
   */
  patArithRandom* addRandomExpression(patArithRandom* p) ;

  /**
   */
  void addMassAtZero(patString aName, patReal aThreshold) ;


  /**
     Build the covariance structure assuming that the indices and ids have alrewady been computed.
   */
  void buildCovariance(patError*& err) ;

  /**
     build the linear utilities containing random parameters.
   */
  void buildLinearRandomUtilities(patError*& err) ;

  /**
     Specifiy wichh random parameter will be modified with the SNP approach
   */
  void setSnpBaseParameter(const patString& n) ;
  /**
     Add a SNP term
   */
  void addSnpTerm(unsigned short i, const patString& n) ;
public:
  /**
   */
  patBoolean automaticScaling ;

  /**
   */
  patString getLatexName(const patBetaLikeParameter* betaParam) ;
  
  /**
   */
  void addOrdinalLogitThreshold(unsigned long id,
				patString paramName) ;

  /**
   */
  void setOrdinalLogitLeftAlternative(unsigned long i) ;

  /**
   */
  unsigned long getOrdinalLogitLeftAlternative() ;

  /**
   */
  map<unsigned long, patBetaLikeParameter*>* getOrdinalLogitThresholds() ;

  /**
   */
  patULong getOrdinalLogitNumberOfIntervals() ;
  /**
   */
  unsigned long getLargestAlternativeUserId() const;
  /**
   */
  unsigned long getLargestGroupUserId() const;
  /**
   */
  void generateCppCodeForAltId(ostream& cppFile,patError*& err) ;
  /**
   */
  void generateCppCodeForGroupId(ostream& cppFile,patError*& err) ;

private:


  patString generateLatexRow(patBetaLikeParameter beta,
			     const patVariables& stdErr,
			     const patVariables& robustStdErr,
			     patReal ttestBasis = 0.0) ;
  void compute_altIdToInternalId() ;
  patString getScaleNameFromId(unsigned long id) const ;
  unsigned long getIdFromScaleName(const patString& scaleName) const ;
  patString getEstimatedParameterName(unsigned long index) const ;
  unsigned long getIndexFromName(const patString& name, patBoolean* success) ;
  //  unsigned long getColumns() ;
  
  // Recursive procedure to generate GNUPLOT commands for the Network GEV model
  ostream& gnuplotNetworkGevNode(patNetworkGevNode* node,
				 ostream& file, 
				 patError*& err) ;



  
protected :
  patModelSpec() ;
  patArithNode* choiceExpr ;
  patArithNode* aggLastExpr ;
  patArithNode* aggWeightExpr ;
  patArithNode* panelExpr ;
  patArithNode* weightExpr ;
  patArithNode* excludeExpr ;
  patArithNode* groupExpr ;
  patModelType modelType ;

  vector<unsigned long> columnData ;

  unsigned long nonFixedBetas ;
  unsigned long nonFixedSigmas ;

  vector<patString>  headers ;
  vector< vector<patString> > headersPerFile ;
  map<patString,patArithNode*> expressions ;
  vector<patString> userOrderOfExpressions ;

  map<patString,patArithVariable*> availExpressions ;   
  map<unsigned long,patString> availName ;

  /**
     The number stored in this map is the index in the attribute
     vector of each individual. It is computed in the computeIndex
     function, when the model is fully loaded.
   */
  map<patString,unsigned long> usedAttributes ;

  vector<patString> attributeNames ;
  map<patString, unsigned long> attribIdPerName ;

  map<unsigned long,unsigned long> altIdToInternalId ;
  map<unsigned long,patString> altIdToName ;
  unsigned long largestAlternativeUserId ;
  unsigned long largestGroupUserId ;
  patBoolean altInternalIdComputed ;
  patBoolean indexComputed ;
  unsigned long sampleEnumeration ;
  map<patString, patAlternative> utilities ;
  map<patString, patAlternative> utilityFormulas ;

  map<unsigned long, patArithNode*> nonLinearUtilities ;
  map<unsigned long, map<patString, patArithNode*> > derivatives ;

  map<patString,patBetaLikeParameter> betaParam ;
  map<patString,patBetaLikeParameter> scaleParam ;
  patBetaLikeParameter mu ;
  map<patString,patNlNestDefinition> nlNestParam ;
  map<patString,patBetaLikeParameter> cnlNestParam ;
  map<patString,patCnlAlphaParameter> cnlAlphaParam ;

  //Logit Kernel

  map<patString,patBetaLikeParameter> logitKernelSigmasParam ;

  // Index = pair(factor,alt)
  // Content = name of the associated sigma parameter
  map<pair<patString,unsigned long>,patString > logitKernelFactors ;


  // Those two maps are built by "check LogitKernel"
  map<patString,unsigned long> factorsIndices ;
  map<unsigned long,list<patString> > factorsPerAlternative ;
  map<patString,patString>  sigmaPerFactor ;

  vector<patBetaLikeParameter*> nonFixedParameters ;
  vector<patReal> scaling ;

  patVariables betaParameters ;
  patReal muValue ;
  patVariables modelParameters ;
  patVariables scaleParameters ;

  patVariables automaticAttributesScales ;
  patVariables automaticParametersScales ;


  patEstimationResult estimationResults ;

  map<patString,pair<patString, patString> > ratios ;
  vector<pair<patString, patString> > constraintNests ;
  vector<patConstantProduct> listOfConstantProducts ;
  
  vector<vector<unsigned long> > nonZeroAlphasPerAlt ;
  vector<vector<unsigned long> > nonZeroAlphasPerNest ;

  // The vector of parameters for CNL is composed of
  // 1) nNests mu_m
  // 2) For each nest, For each alt such that alpha(nest,alpha) !=0, the
  // corresponding alpha
  // The value cumulIndex[i] contains the index in the vector of parameters of the first alpha corresponding to nest i, i=0,...nNest-1
  // The value cumulIndex[nNest] contains the total number of parameters

  vector<unsigned long> cumulIndex ;

  // Store the iterators as they are called for efficiency sake
  vector<patStlVectorIterator<unsigned long>*> iterPerNest ;
  vector<patStlVectorIterator<unsigned long>*> iterPerAlt ;

  map<patString, patBetaLikeParameter> networkGevNodes ;
  map<patString, patNetworkGevLinkParameter> networkGevLinks ;

  patNetworkGevModel* theNetworkGevModel ;

  std::set<patString> nodesInGnuplotFile ;

  patListLinearConstraint listOfLinearConstraints ;

  patBoolean logitKernelChecked ;

  unsigned long nonZeroEntriesInFactorLoading ;
  
  patListNonLinearConstraints* equalityConstraints ;
  patListNonLinearConstraints* inequalityConstraints ;

  map<patString,pair<patRandomParameter,patArithRandom*> > randomParameters ;
  map<pair<patString,patString>,patBetaLikeParameter*> covarParameters ;

  // For each random coefficient, stores the estimator its variance,
  // and the std error of the estimator
  map<patString,pair<patReal,patReal> > varianceRandomCoef ;
  map<patString,patBoolean>  relevantVariance ;

  // For each pair of random coefficients, stores the estimator their
  // covariance, and the std error of the estimator
  map<pair<patString,patString>, pair<patReal,patReal> > covarianceRandomCoef ;
  map<pair<patString,patString>,patBoolean > relevantCovariance ;

  unsigned long numberOfDraws ;
  unsigned long algoNumberOfDraws ;
  unsigned long nbrDrawAttributesPerObservation ;
  // List of parameters to include in the summary
  vector<patString> summaryParameters ;

  // Type of distribution for each index in the draws

  vector<patDistribType> typeOfDraws ;
  vector<patBoolean> areDrawsPanel ;
  vector<patReal> massAtZero ;

  // List of random parameter varying with observations
  vector<patRandomParameter*> observationsParameters ;

  // List of random parameter varying with individual in panel data
  vector<patRandomParameter*> individualParameters ;

  // List of random parameters with a discrete distribution
  vector<patDiscreteParameter> discreteParameters ;

  list<patString> panelVariables ;
  list<patString> modelDescription ;
  
  list<pair<patString,patReal> > listOfMassAtZero ;

  patString snpBaseParameter ;
  list<pair<unsigned short,patString> > listOfSnpTerms ;
  vector<unsigned short> ordersOfSnpTerms ;
  vector<patBetaLikeParameter*> coeffOfSnpTerms ;
  vector<unsigned long> idOfSnpBetaParameters ;
  patIterator<patBetaLikeParameter>* allBetaIter ;
  
  map<unsigned long, patString> selectionBiasParameters ;
  vector<patBetaLikeParameter*> selectionBiasPerAlt ;
  
  // Parameter which varies in gnuplot, and its bounds
  patBoolean gnuplotDefined ;
  patString gnuplotParam ;
  patReal gnuplotMin ;
  patReal gnuplotMax ;

  // gradient

  patBoolean firstGatherGradient ;
  patVariables thisGradient ;
  vector<unsigned long> keepBetaIndices ; 
  patBoolean estimGroupScale ;

  patFormatRealNumbers theNumber ;

  map<patString,patString> latexNames ;

  map<unsigned long, patString> ordinalLogitThresholds ;
  map<unsigned long, patBetaLikeParameter*> ordinalLogitBetaThresholds ;
  unsigned long ordinalLogitLeftAlternative ;

  // ZhengFosgerau

  vector<patOneZhengFosgerau> zhengFosgerau;
  unsigned short numberOfProbaInZf ;

public:

  // List of group ids
  list<long> groupIds ;

  patBoolean allAlternativesAlwaysAvail ;


  // Description of the model for N*****/Gianluca project
  patString regressionObservation ;
  patString fixationStartingTime ;
  patUtilFunction acquisitionModel ;
  patUtilFunction validationModel ;
  patString sigmaAcqName ;
  patBetaLikeParameter* sigmaAcq ;
  patString sigmaValName ;
  patBetaLikeParameter* sigmaVal ;
  patString durationNameParam ;
  patBetaLikeParameter* durationModelParam ;
  unsigned long startingTimeId ;
  unsigned long gianlucaObservationId ;
  patBoolean useModelForGianluca ;
  /**
   */
  patError* syntaxError ;

  /**
   */
  patIterator<patBetaLikeParameter>* betaIterator ;
  patIterator<patBetaLikeParameter>* scaleIterator ;
  patIterator<patBetaLikeParameter>* nlNestIterator ;
  patIterator<patBetaLikeParameter>* cnlNestIterator ;
  patIterator<patBetaLikeParameter>* cnlAlphaIterator ;
  patIterator<patCnlAlphaParameter>* cnlFullAlphaIterator ;
  patIterator<patBetaLikeParameter>* networkNestIterator ;
  patIterator<patBetaLikeParameter>* networkAlphaIterator ;
  patSequenceIterator<patBetaLikeParameter> allBetaIterator ; 
  patSequenceIterator<patBetaLikeParameter> allModelIterator ; 
  patSequenceIterator<patBetaLikeParameter> allParametersIterator ; 

  map<patString,  list<long> > iiaTests ;
  map<pair<patString,patString>,patReal> probaStandardErrors ;
  


  patBetaLikeParameter* generalizedExtremeValueParameter ;
  patString generalizedExtremeValueParameterName ;

};

#endif

