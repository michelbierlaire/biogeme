//-*-c++-*------------------------------------------------------------
//
// File name : patSample.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jul 11 22:25:18 2000
//
//--------------------------------------------------------------------

#ifndef patSample_h
#define patSample_h

#include <map>
#include <list>
#include "patIterator.h"
#include "patError.h"
#include "patLegendre.h"
#include "patVariables.h"

class patLikelihood ;
class patModelSpec ;
class patRandomNumberGenerator ;
class patDataStorage ;
class patIndividualData ;
class patAggregateObservationData ;
class patObservationData ;

/**
   @doc This object is in charge of the sample management.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Jul 11 22:25:18 2000)
 */
class patSample {

public :

  /**
     Sole constructor
   */
  patSample() ;

  /**
     Dtor
   */
  ~patSample() ;


  /**
     Process data from an external data structure
   */
  void externalDataStructure(patPythonReal** array, unsigned long nr, unsigned nc) ;
  
  /**
     Generate the attributes used for model estimation from the data
     file. Also, generate random numbers for mixed logit estimation.
     @param rndNumbers pointer to the random number generator
     @param err pointer to the error object
   */
  void readDataFile(patRandomNumberGenerator* normalRndNumbers,
		    patRandomNumberGenerator* unifRndNumbers,
		    patError*& err) ;
  

  /**
   */
  void processData(patRandomNumberGenerator* normalRndNumbers,
		   patRandomNumberGenerator* unifRndNumbers,
		   patIterator<pair<vector<patString>*,vector<patReal>*> >* theIterator,
		   patString dataName,
		   patError*& err)  ;
  /**
   */
  unsigned long getSampleSize() const ;

  /**
   */
  unsigned long getNumberOfIndividuals() const ;

  /**
   */
  unsigned long getNumberOfObservations()  ;

  /**
   */
  unsigned long getNumberOfAggregateObservations()  ;
  
  /**
   */
  patIterator<pair<vector<patString>*,vector<patReal>* > >* 
  createFileIterator(patString fileName,
		     unsigned long dataPerRow,
		     vector<patString>* headers) ;
  
  /**
   */
  patIterator<patObservationData*>* createObsIterator()
 ;

  /**
     This function creates a vector of iterators, each one taking care
     of a different portion of the sample, so that they can be
     associated with different threads.
   */
  vector<patIterator<patObservationData*>* > 
  createObsIteratorThread(unsigned int nbrThreads, 
			  patError*& err) ;

  /**
     This function creates a vector of iterators, each one taking care
     of a different portion of the sample, so that they can be
     associated with different threads.
   */
  vector<patIterator<patIndividualData*>* > 
  createIndIteratorThread(unsigned int nbrThreads, 
			  patError*& err) ;

  /**
   */
  patIterator<patAggregateObservationData*>* createAggObsIterator()
 ;

  /**
   */
  patIterator<patIndividualData*>* createIndIterator() ;


  /**
     @return Group indices are numbered from 0 to nGroup-1, while it is not 
             required in the data files. This function provides the index, 
             given the group label in the data file.
  */
  unsigned long groupIndex(unsigned long)  ; 
  /**
   */
  unsigned long numberOfGroups() const ;

  /**
   */
  void empty() ;

  /**
   */
  void generateSimulatedData(patLikelihood* like,
			     patError*& err) ;

  /**
   */
  void generateChoice(patLikelihood* like,
		      patError*& err) ;


  /**
   */
  void shuffleSample() ;


  /**
    The number of cases is the sum of the number of alternatives
    available to each observation minus the number of observations.
   */
  unsigned long getCases() const ;

  /**
     Returns the number of rows in the file contaninig data, that is
     without counting thew first row with labels.
   */
  unsigned long getNbrOfDataRowsInFile() ;
  /**
     Provides a power of 10 with the same level of magnitude as the
     mean value of the attributes
   */
  patVariables* getLevelOfMagnitude(patError*& err) ;
  /**
   */
  void scaleAttributes(patVariables* scsles, patError*& err) ;

  /**
   */
  unsigned long nbrObsForPerson(unsigned long i) ;
  
  /**
     Compute the loglikelihood of a model with constants only and all
     alternatives available. The value is
     sum_j n_j ln n_j - n ln n
   */
  patReal computeLogLikeWithCte() ;

private :
  unsigned long nAttributes ;
  unsigned long nAlternatives ;
  unsigned long cases ;
  unsigned long nDataRowsInFile ;
  unsigned long numberOfObservations ;
  unsigned long numberOfAggregateObservations ;
  patPythonReal** dataArrayProvidedByUser ;
 patIterator<patObservationData*>* theObsPtr ;


  patLegendre legendrePolynomials ;
  vector<patString> headers ;
  patDataStorage* warehouse ;
  map<long,long> group ;
  map<long,patReal> weightedGroup ;
  vector<long> grIndex ;



  map<patString,unsigned long> numberOfAttributes ;
  map<patString,patReal> meanOfAttributes ;
  map<patString,patReal> minOfAttributes ;
  map<patString,patReal> maxOfAttributes ;

  vector<patReal> levelOfMagnitude ;

  
  map<unsigned long, unsigned long> nbrOfObsPerIndividual ;

  map<unsigned long, unsigned long> availableAlt ;
  map<unsigned long, unsigned long> chosenAlt ;
  map<unsigned long, patReal> weightedChosenAlt ;


  patReal totalWeight ;


  patBoolean panelDataFeature ;
  unsigned long nRows ;
  unsigned long nColumns ;



  
};

#endif
