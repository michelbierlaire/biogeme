
//-*-c++-*------------------------------------------------------------
//
// File name : bioSample.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jun 19 10:35:31  2009
//
//--------------------------------------------------------------------


#include <set>
#include <iostream>
#include <fstream>
#include "patFileNames.h"
#include "bioSample.h"
#include "bioParameters.h"
#include "patDisplay.h"
#include "patOutputFiles.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "bioIteratorInfoRepository.h"
#include "bioMetaIterator.h"
#include "bioRowIterator.h"
#include "bioIdIterator.h"
#include "bioLiteralRepository.h"
#include "bioLiteralValues.h"
#include "bioExpression.h"
#include "patTimeInterval.h"
#include "patFileExists.h"

bioSample::bioSample(patString n, patError*& err) : fileName(n) {
  // Read the headers
  // Open the file

  patString line ;
  patString str ;
  ifstream in(fileName.c_str());
  if (!in || in.fail())
  {
    stringstream str ;
    str << "Cannot open the file " << fileName ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }

  // The first header is the row id

  patString rowIdHeader =  bioParameters::the()->getValueString(patString("HeaderOfRowId"),err) ; 
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  headers.push_back(rowIdHeader) ;
  // The column ID is set to patBadId. An virtualcolumn ID will be
  // assigned by the literal repository.
  bioLiteralRepository::the()->addVariable(rowIdHeader,patBadId,err) ;


  // Read headers
  patULong count = 1 ;
  getline(in, line) ;
  istringstream iss(line);
  while (iss >> str) {
    this->headers.push_back(str) ;
    bioLiteralRepository::the()->addVariable(str,count,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    ++count ;
  }

  numberOfRealColumns = count ;
  //  DETAILED_MESSAGE("There are " << numberOfRealColumns << " columns in the file, including a column containing the row number, with header " << rowIdHeader) ;
  

  // DETAILED_MESSAGE("Headers from the data file:") ;
  // printHeaders() ;
  in.close() ;

  generatePythonHeaders(err) ;
   if (err != NULL) {
     WARNING(err->describe()) ;
     return ;
   }

}

bioSample::~bioSample() {
}

void bioSample::readFile(bioExpression* exclude, 
			 vector<pair<patString, bioExpression*> >* userExpressions, 
			 patError*& err) {
  patString binaryFile = getBinaryFileName() ;
  if (patFileExists()(binaryFile)) {
    readBinaryFile(exclude, 
		   userExpressions, 
		   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  else {
    readTextFile(exclude, 
		 userExpressions, 
		 err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
      
  }
}

void bioSample::readTextFile(bioExpression* exclude, 
			 vector<pair<patString, bioExpression*> >* userExpressions, 
			 patError*& err) {
  
  dataReadFromBinary = patFALSE ;
  ofstream binFile;
  patString binaryFile = getBinaryFileName() ;
  binFile.open (binaryFile.c_str(), ios::out | ios::binary);

  // The binary file has the following structure.
  // First: patUlong: number of columns
  // Then, each row is coded as a vector of patReal

  patAbsTime beg ;
  beg.setTimeOfDay() ;
  processTime.setStart(beg) ;
  DEBUG_MESSAGE("***** " << fileName << " *****") ;
  // only to display total  line number while reading
  ifstream f(fileName.c_str());
  patString l;
  int nbLines=0;
  while(std::getline(f, l)){
    nbLines++;
  }

  patString rowIdHeader =  bioParameters::the()->getValueString(patString("HeaderOfRowId"),err) ; 
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  headers.push_back(rowIdHeader) ;

  for (vector<pair<patString, bioExpression*> >::iterator i = userExpressions->begin() ;
       i != userExpressions->end() ;
       ++i) {
    headers.push_back(i->first) ;
  }

  patULong nbrOfColumns = headers.size() ;
  binFile.write((char*)&numberOfRealColumns,sizeof(patULong)) ;

  DEBUG_MESSAGE("NUMBER OF REAL COLUMNS: " << numberOfRealColumns) ;

  ifstream in(fileName.c_str());
  if (!in || in.fail()) {
    stringstream str ;
    str << "Cannot open the file " << fileName ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  // Skip headers
  patString line ;
  getline(in, line) ;


  GENERAL_MESSAGE("Read sample file: " << fileName) ;

  rowNumber = 0.0 ;
  physicalRowNumber = 0 ;
  bioLiteralValues::the()->eraseValues() ;

  DEBUG_MESSAGE("MEMORY FOR " << nbrOfColumns << " COLUMNS") ;
  vector<patReal> listOfV(nbrOfColumns) ;
  
  nbrOfExcludedRows = 0 ;

  while(in && !in.eof()) {
    ++physicalRowNumber ;

    //fprintf(stdout, "%c [%5lu/%i]", chars[physicalRowNumber % sizeof(chars)], physicalRowNumber, nbLines);
    //fflush(stdout);
    
    listOfV[0] = rowNumber ;
    binFile.write((char*)&(listOfV[0]),sizeof(patReal)) ;
    in >> listOfV[1] ;
    binFile.write((char*)&(listOfV[1]),sizeof(patReal)) ;
    if (!in.eof()) {
      for (unsigned long i = 2 ;  i < numberOfRealColumns ; ++i) {
	if (!in.eof()) {
	  in >> listOfV[i] ;
	  binFile.write((char*)&(listOfV[i]),sizeof(patReal)) ;
	}
      }
      processRow(exclude,listOfV,userExpressions,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  } 
  // Close file
  in.close() ;
  patAbsTime endc ;
  endc.setTimeOfDay() ;
  processTime.setEnd(endc) ;
  GENERAL_MESSAGE("Processing the sample file: " << fileName << " ["<<processTime.getLength()<<"]") ;

  binFile.close() ;
  patOutputFiles::the()->addDebugFile(binaryFile,"Data file in a binary format");
  
}


void bioSample::readBinaryFile(bioExpression* exclude, 
			 vector<pair<patString, bioExpression*> >* userExpressions, 
			 patError*& err) {
  
  dataReadFromBinary = patTRUE ;
  patAbsTime beg ;
  beg.setTimeOfDay() ;
  processTime.setStart(beg) ;
  patString binaryFile = getBinaryFileName() ;
  ifstream binFile;
  binFile.open (binaryFile.c_str(), ios::in | ios::binary);

  GENERAL_MESSAGE("Read sample in binary format from " << binaryFile << " *****") ;
  // only to display total  line number while reading

  patString rowIdHeader =  bioParameters::the()->getValueString(patString("HeaderOfRowId"),err) ; 
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  headers.push_back(rowIdHeader) ;
  for (vector<pair<patString, bioExpression*> >::iterator i = userExpressions->begin() ;
       i != userExpressions->end() ;
       ++i) {
    headers.push_back(i->first) ;
  }
  patULong nbrOfBinaryColumns ;
  binFile.read((char*)&nbrOfBinaryColumns,sizeof(patULong)) ;
  DEBUG_MESSAGE("NUMBER OF COLUMNS: " << nbrOfBinaryColumns) ;

  if (nbrOfBinaryColumns != numberOfRealColumns) {
    stringstream str ;
    str << "The binary file " << binaryFile << " contains " << nbrOfBinaryColumns << " and the text file " << fileName << " contains " << numberOfRealColumns << "columns. Erase the file " << binaryFile << " and start again." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  rowNumber = 0.0 ;
  physicalRowNumber = 0 ;
  bioLiteralValues::the()->eraseValues() ;
  // Prepare the memory for all columns, but read only from the file
  // the "real" ones. The others will be generated during the
  // processing
  DEBUG_MESSAGE("MEMORY FOR " << getNumberOfColumns() << " COLUMNS") ;
  vector<patReal> listOfV(getNumberOfColumns(),0.0) ;
  
  nbrOfExcludedRows = 0 ;

  while(binFile && !binFile.eof()) {
    ++physicalRowNumber ;
    for (unsigned long i = 0 ;  i < numberOfRealColumns ; ++i) {
      binFile.read((char*)&(listOfV[i]),sizeof(patReal)) ;
    }
    if (!binFile.eof()) {
      processRow(exclude,listOfV,userExpressions,err) ;
      if (err != NULL) {
	stringstream str ;
	str << " Problem detected while reading row " << physicalRowNumber << " of " << fileName ;
	err->addComment(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  // Close file
  binFile.close() ;
  patAbsTime endc ;
  endc.setTimeOfDay() ;
  processTime.setEnd(endc) ;
  GENERAL_MESSAGE("Processing the sample file: " << binaryFile << " ["<<processTime.getLength()<<"]") ;
  
}


patULong bioSample::size() const {
  return samples.size() ;
}


vector<patString> bioSample::getHeaders() const {
  return headers ;
}


patVariables* bioSample::at(patULong index) {
  if (index >= samples.size()) {
    return NULL ;
  }
  return &(samples[index]) ;
}


patULong bioSample::getIndexFromHeader(patString h,patError*& err) const {
  patULong result =  bioLiteralRepository::the()->getColumnIdOfVariable(h,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patBadId ;
  }
  return result ;
}


patString bioSample::getTextFileName() const {
  return fileName ;
}


patBoolean bioSample::isEqual(bioSample s) {
  return s.getTextFileName() == this->fileName ;
}


void bioSample::printSample() {
  DEBUG_MESSAGE("*****************") ;
  printHeaders() ;
  patULong cpt = 0;
  for (vector<patVariables>::const_iterator it=samples.begin() ;
       it != samples.end() ;
       it++ ) {
    std::cout << "[" << cpt++ << "] " << *it << endl ;
  }
  std::cout << std::endl ;
}


void bioSample::addObservationData(patVariables data) {
  samples.push_back(data) ;
}


void bioSample::printHeaders() {
  for (vector<patString>::const_iterator it=headers.begin() ;
       it != headers.end();
       it++ ) {
     std::cout << *it << "    ";
  }
  std::cout << std::endl ;
}


void bioSample::computeMapOfDatabase(patError*& err) {

  vector<patString> listOfIterators = 
    bioIteratorInfoRepository::the()->getListOfIterators() ;

  map<patULong, patReal> currentValue ;
  
  for (vector<patString>::iterator i = listOfIterators.begin() ;
       i != listOfIterators.end() ;
       ++i) {
    bioIteratorType theType = bioIteratorInfoRepository::the()->getType(*i,err) ;  if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    if (theType == ROW || theType == META) {
      patString indexName =  bioIteratorInfoRepository::the()->getIndexName(*i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patULong colId = getIndexFromHeader(indexName,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (colId == patBadId) {
	stringstream str ;
	str << "Column " << indexName << " characterizing iterator " << *i << " is unknown in file " << fileName ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      currentValue[colId] = patReal() ;
    }
  }

  for (patULong row = 0 ;
       row < samples.size() ;
       ++row) {
    if (row == 0) {
      for (map<patULong, patReal>::iterator col = currentValue.begin() ;
	   col != currentValue.end() ;
	   ++col) {
	bioIteratorInfoRepository::the()->addNewRowId(headers[col->first],0) ;
        col->second = samples[0][col->first] ;
      }
    }
    else {
      for (map<patULong, patReal>::iterator col = currentValue.begin() ;
	   col != currentValue.end() ;
	   ++col) {
	if (col->second != samples[row][col->first]) {
	  bioIteratorInfoRepository::the()->addNewRowId(headers[col->first],row) ;
	  col->second = samples[row][col->first] ;
	}
      }
    }
  }

  bioIteratorInfoRepository::the()->computeRowPointers(samples.size(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}

bioMetaIterator* bioSample::createMetaIterator(bioIteratorSpan theSpan, bioIteratorSpan theThreadSpan, patError*& err) const {
  
  //  DEBUG_MESSAGE("CREATE META ITERATOR FOR THREAD " << theThreadSpan) ;

  if (theSpan.lastRow == patBadId) {
    // theSpan.lastRow = size()-1 ;
    theSpan.lastRow = size() ;
  }
  bioMetaIterator* ptr = new bioMetaIterator(&samples,
					     theSpan, 
					     theThreadSpan,
					     err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }
  return ptr ;
}

bioRowIterator* bioSample::createRowIterator(bioIteratorSpan theSpan, bioIteratorSpan theThreadSpan, patBoolean printMessages, patError*& err) const {
  
  if (theSpan.lastRow == patBadId) {
    // theSpan.lastRow = size()-1 ;
    theSpan.lastRow = size() ;
  }
  bioRowIterator* ptr = new bioRowIterator(&samples,theSpan,theThreadSpan,printMessages,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  return ptr ;
}


bioRowIterator* bioSample::createIdIterator(bioIteratorSpan theSpan, bioIteratorSpan theThreadSpan, patString header, patError*& err) const {

  if (theSpan.lastRow == patBadId) {
    // theSpan.lastRow = size()-1 ;
    theSpan.lastRow = size() ;
  }

  patULong indexId = getIndexFromHeader(header,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (indexId == patBadId) {
    stringstream str ;
    str << "Header " << header << " is unknown" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  DEBUG_MESSAGE("Create an iterator on column " << indexId << ": " << header) ;
  bioIdIterator* ptr = new bioIdIterator(&samples,theSpan,theThreadSpan,indexId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  return ptr ;
  
}


void bioSample::prepare(bioExpression* exclude, 
			vector<pair<patString, bioExpression*> >* userExpressions, 
			patError*& err) {
  DEBUG_MESSAGE("Prepare data...") ;

  readFile(exclude,userExpressions,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  computeMapOfDatabase(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}

void bioSample::generatePythonHeaders(patError*& err) {
  ofstream f("headers.py") ;
  f << "from biogeme import *" << endl ;
  for (vector<patString>::iterator i = headers.begin() ;
       i != headers.end() ;
       ++i) {
    f << *i << "=Variable('" << *i << "')" << endl ;
  }
  f.close() ;
  patOutputFiles::the()->addDebugFile("headers.py","List of headers of the data file (biogeme use only)");
}

vector<patULongLong> bioSample::getListOfIds(patString header, patError*& err) {
  patULong indexId = getIndexFromHeader(header,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patULongLong>() ;
  }
  if (indexId == patBadId) {
    stringstream str ;
    str << "Header " << header << " is unknown" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return vector<patULongLong>() ;
  }
  set<patULongLong> listOfUniqueIds ;
  for (vector<patVariables>::iterator row = samples.begin();
       row != samples.end() ;
       ++row) {
    listOfUniqueIds.insert(patULongLong((*row)[indexId])) ;
  }
  DETAILED_MESSAGE("Variable " << header << " has " << listOfUniqueIds.size() << " different values in the sample") ;
  vector<patULongLong> result ;
  for (set<patULongLong>::iterator i = listOfUniqueIds.begin() ;
       i != listOfUniqueIds.end() ;
       ++i) {
    result.push_back(*i) ;
  }
  return result ;
}

patULong bioSample::getNumberOfColumns() const {
  return headers.size() ;
}


void bioSample::setColumn(patString column, 
			  vector<patReal> values, 
			  patBoolean duplicate, 
			  patString individual,
			  patError*& err) {

  patULong theColumn = 
    bioLiteralRepository::the()->getColumnIdOfVariable(column,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patULong theId = 
    bioLiteralRepository::the()->getColumnIdOfVariable(individual,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patReal currentIndividual = patReal(patBadId) ;
  patULong currentIndex = 0 ;
  for ( patULong row = 0 ;
       row < samples.size() ;
       ++row) {
    if (samples[row][theId] != currentIndividual) {
      if (currentIndex >= values.size()) {
	err = new patErrOutOfRange<patULong>(currentIndex,0,values.size()-1) ;
	WARNING(err->describe()) ;
	return ;
      }
      samples[row][theColumn] = values[currentIndex] ;
      ++currentIndex ;
      currentIndividual = samples[row][theId] ;
    }
    else {
      if (duplicate) {
	samples[row][theColumn] = values[currentIndex] ;
      }
    }
  }
}

vector<patReal> bioSample::getColumn(patString column, patString individual, patError*& err) {

  vector<patReal> result ;
  patULong theColumn = 
    bioLiteralRepository::the()->getColumnIdOfVariable(column,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patReal>();
  }

  patULong theId = 
    bioLiteralRepository::the()->getColumnIdOfVariable(individual,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patReal>();
  }

  patReal currentIndividual = patReal(patBadId) ;
  //  patULong currentIndex = 0 ;
  for ( patULong row = 0 ;
       row < samples.size() ;
       ++row) {
    if (samples[row][theId] != currentIndividual) {
      result.push_back(samples[row][theColumn]) ;
      currentIndividual = samples[row][theId] ;
    }
  }
  return result ;
}


patULong bioSample::getNbrExcludedRow() const {
  return nbrOfExcludedRows ;
}

const patVariables* bioSample::getFirstRow() {
  if (samples.empty()) {
    return NULL ;
  }
  return &(samples[0]) ;
}

void bioSample::processRow(bioExpression* exclude, 
			   vector<patReal> listOfV, 
			   vector<pair<patString, bioExpression*> >* userExpressions, 
			   patError*& err)  {

  // Include user expressions
  if (userExpressions != NULL) {
    for (vector<pair<patString, bioExpression*> >::iterator i = 
	   userExpressions->begin() ;
	 i != userExpressions->end() ;
	 ++i) {

      i->second->setVariables(&listOfV) ;
      patReal value = i->second->getValue(patFALSE, patLapForceCompute, err) ;   
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patULong col = 
	bioLiteralRepository::the()->getColumnIdOfVariable(i->first,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      listOfV[col] = value ;
    }
  }
  patReal ex ;
  if (exclude != NULL) {
    exclude->setVariables(&listOfV) ;
    ex = exclude->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  else {
    ex = 0 ;
  }

  if (ex == 0) {
    rowNumber += 1.0 ;
    samples.push_back(listOfV) ;
  }
  else {
    ++nbrOfExcludedRows ;
  }

}

patString bioSample::getProcessingTime() const {
  return processTime.getLength() ;
}

patString bioSample::getBinaryFileName() const {

  stringstream str ;
  str << "__bin_" << extractFileNameFromPath(fileName) ;
  return patString(str.str()) ;
}
patString bioSample::getDataFileName() const {
  if (dataReadFromBinary) {
    return getBinaryFileName() ;
  }
  else {
    return getTextFileName() ;
  }
}
