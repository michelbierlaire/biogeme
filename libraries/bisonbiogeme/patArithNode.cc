//-*-c++-*------------------------------------------------------------
//
// File name : patArithNode.cc
// Author :    Michel Bierlaire
// Date :      Wed Nov 22 17:01:57 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patErrNullPointer.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patArithNode.h"
#include "patArithAttribute.h"

patArithNode::patArithNode(patArithNode* par,
			   patArithNode* left, 
			   patArithNode* right) : 
  leftChild(left),
  rightChild(right),
  parent(par) {

}

patArithNode::~patArithNode() {
  if (leftChild != NULL) {
    DELETE_PTR(leftChild) ;
  }
  if (rightChild != NULL) {
    DELETE_PTR(rightChild) ;
  }
}

patBoolean patArithNode::isTop() const {
  return (parent == NULL) ; 
}

patArithNode* patArithNode::getLeftChild() const {
  return leftChild ;
}

patArithNode* patArithNode::getRightChild() const {
  return rightChild ;
}

patArithNode* patArithNode::getParent() const {
  return parent ;
}

patString patArithNode::getExpression(patError*& err) const {

  patString result ;
  patString leftResult ;
  patString rightResult ;
  switch (getOperatorType()) {
  case BINARY_OP :
    {
      if (leftChild == NULL || rightChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patString();
      }

      leftResult = leftChild->getExpression(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
      rightResult = rightChild->getExpression(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
    
      result = "" ;
      if (leftChild->getOperatorType() == BINARY_OP) {
	result += "(" ;
      }
      result += " " ;
      result += leftResult ;
      result += " " ;
      if (leftChild->getOperatorType() == BINARY_OP) {
	result += ")" ;
      }
      result += " " ;
      result += getOperatorName() ;
      result += " " ;
      if (rightChild->getOperatorType() == BINARY_OP) {
	result += "(" ;
      }
      result += " " ;
      result += rightResult ;
      result += " " ;
      if (rightChild->getOperatorType() == BINARY_OP) {
	result += ")" ;
      }
      return result ;
    }    
  case UNARY_OP :
    {
      if (leftChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patString();
      }

      leftResult = leftChild->getExpression(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
    
      result =  getOperatorName() ;
      result += '(' ;
      result += leftResult ;
      result += ')' ;
      return result ;
    }

  case VARIABLE_OP :
    {
      patString res(getOperatorName()) ;
      res += " " ;
      return res ;
    }
  case ATTRIBUTE_OP :
    {
      patString res(getOperatorName()) ;
      res += " " ;
      return res ;
    }

  case CONSTANT_OP :
    {
      stringstream str ;
      str << getValue(err) ;
      return patString(str.str()) ;
      break ;
    }

  default:
    {
    err = new patErrMiscError("Operator not defined");
    return patString();
    }
  }
}


void patArithNode::setParent(patArithNode* par) {
//   if (parent != NULL) {
//     WARNING("Parent already defined. New value overwrites old one.");
//   }
  parent = par ;
}

ostream& operator<<(ostream& str, const patArithNode& x) {
  patError* err = NULL ;
  str << x.getExpression(err) ;
  if (err != NULL) {
    str << err->describe() << '\t' ;
  }
  return str ;
}


patString patArithNode::getGnuplot(patError*& err) const {
  patString result ;
  patString leftResult ;
  patString rightResult ;
  switch (getOperatorType()) {
  case BINARY_OP :
    {
      if (leftChild == NULL || rightChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patString();
      }

      leftResult = leftChild->getGnuplot(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
      rightResult = rightChild->getGnuplot(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
    
      result = "" ;
      if (leftChild->getOperatorType() == BINARY_OP) {
	result += "(" ;
      }
      result += " " ;
      result += leftResult ;
      result += " " ;
      if (leftChild->getOperatorType() == BINARY_OP) {
	result += ")" ;
      }
      result += " " ;
      result += getOperatorName() ;
      result += " " ;
      if (rightChild->getOperatorType() == BINARY_OP) {
	result += "(" ;
      }
      result += " " ;
      result += rightResult ;
      result += " " ;
      if (rightChild->getOperatorType() == BINARY_OP) {
	result += ")" ;
      }
      return result ;
    }    
  case UNARY_OP :
    {
      if (leftChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patString();
      }

      leftResult = leftChild->getGnuplot(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
    
      result =  getOperatorName() ;
      result += '(' ;
      result += leftResult ;
      result += ')' ;
      return result ;
    }

  case VARIABLE_OP :
    {
      patString res(getOperatorName()) ;
      return res ;
    }

  case ATTRIBUTE_OP :
    {
      patString res(getOperatorName()) ;
      res += "(x) " ;
      return res ;
    }

  case CONSTANT_OP :
    {
      stringstream str ;
      str << getValue(err) ;
      return patString(str.str()) ;
      break ;
    }

  default:
    {
      stringstream str ;
      str << *this ;
      return patString(str.str()) ;
    }
  }

}

void patArithNode::setVariable(const patString& s, unsigned long i) {
  if (leftChild != NULL) {
    leftChild->setVariable(s,i) ;
  }
  if (rightChild != NULL) {
    rightChild->setVariable(s,i) ;
  }
}

patArithNode* patArithNode::getRoot() const {
  if (isTop()) {
    return (patArithNode*)this ;
  }
  else {
    return getParent()->getRoot() ;
  }
}

patString patArithNode::getInfo() const {
  stringstream str ;
  if (isTop()) {
    str << "[" << *this << "]"  ;
    return patString(str.str());
  }
  str << "[" << *this << "] within [" << *getRoot() << "]" ;
  return patString(str.str()) ;
}


vector<patString>* patArithNode::getLiterals(vector<patString>* listOfLiterals,
					     vector<patReal>* valuesOfLiterals,
					     patBoolean withRandom,
					     patError*& err) const {
  
  if (err != NULL) {
    return NULL ;
  }

  if (listOfLiterals == NULL) {
    err = new patErrNullPointer("vector<patString>") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  patArithNode* leftChild ;
  patArithNode* rightChild ;
  switch (getOperatorType()) {
    
  case UNARY_OP:
    leftChild = getLeftChild() ;
    if (leftChild != NULL) {
      listOfLiterals = leftChild->getLiterals(listOfLiterals,
					      valuesOfLiterals,
					      withRandom,
					      err) ;
    }
    return listOfLiterals ;
    break ;
  case BINARY_OP:
    leftChild = getLeftChild() ;
    if (leftChild != NULL) {
      listOfLiterals = leftChild->getLiterals(listOfLiterals,
					      valuesOfLiterals,
					      withRandom,
					      err) ;
    }
    rightChild = getRightChild() ;
    if (rightChild != NULL) {
      listOfLiterals = rightChild->getLiterals(listOfLiterals,
					       valuesOfLiterals,
					       withRandom,
					       err) ;
    }
    return listOfLiterals ;
    break ;
  case UNIRANDOM_OP:
  case RANDOM_OP:
    err = new patErrMiscError("This function should not be called for random variables") ;
    WARNING(err->describe()) ;
    return listOfLiterals ;
  case VARIABLE_OP:
  case ATTRIBUTE_OP:
    listOfLiterals->push_back(getOperatorName()) ;
    if (valuesOfLiterals != NULL) {
      patReal val = getValue(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      valuesOfLiterals->push_back(val) ;
    }
    return listOfLiterals ;

  case CONSTANT_OP:
    return listOfLiterals ;
  default:
    err = new patErrMiscError("Unknown node type") ;
    WARNING(err->describe()) ;
    return NULL ;
    break ;
  }
}

void patArithNode::setAttribute(const patString& s, unsigned long i) {
  if (leftChild != NULL) {
    if (leftChild->getOperatorType() == VARIABLE_OP && leftChild->getOperatorName() == s) {
      
      patArithAttribute* ptr = new patArithAttribute(this) ;
      ptr->setName(leftChild->getOperatorName()) ;
      ptr->setId(i) ;
      DELETE_PTR(leftChild) ;
      leftChild = ptr ;
    }
    else {
      leftChild->setAttribute(s,i) ;
    }
  }
  if (rightChild != NULL) {
    if (rightChild->getOperatorType() == VARIABLE_OP && rightChild->getOperatorName() == s) {
      patArithAttribute* ptr = new patArithAttribute(this) ;
      ptr->setName(rightChild->getOperatorName()) ;
      ptr->setId(i) ;
      DELETE_PTR(rightChild) ;
      rightChild = ptr ;
    }
    else {
      rightChild->setAttribute(s,i) ;
    }
  }
}

ostream& patArithNode::printLiterals(ostream& str,patError*& err) const  {
  vector<patString> literals ;
  vector<patReal> values ;
  getLiterals(&literals,&values,patTRUE,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return str ;
  }
  if (literals.size() != values.size()) {
    stringstream s ;
    s << "Incompatible sizes: " <<  literals.size() << " and " << values.size() ;
    err = new patErrMiscError(s.str()) ;
    WARNING(err->describe());
    return str ;
  }
  for (unsigned long i = 0 ; i < literals.size() ; ++i) {
    str << endl << literals[i] << "=" << values[i]  ;
  }
  str << endl ;
  return str ;
}

void patArithNode::replaceInLiterals(patString subChain, patString with) {
  if (leftChild != NULL) {
    leftChild->replaceInLiterals(subChain,with) ;
  }
  if (rightChild != NULL) {
    rightChild->replaceInLiterals(subChain,with) ;
  }
}


void patArithNode::expand(patError*& err) {
  if (leftChild != NULL) {
    leftChild->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
    
  }
  if (rightChild != NULL) {
    rightChild->expand(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  
  
}

void patArithNode::computeParamId(patError*& err) {
  if (leftChild != NULL) {
    leftChild->computeParamId(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  if (rightChild != NULL) {
    rightChild->computeParamId(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}



patBoolean patArithNode::isDerivativeStructurallyZero(unsigned long index, patError*& err) {
  map<unsigned long,patBoolean>::iterator found = derivStructurallyZero.find(index) ;
  if (found != derivStructurallyZero.end()) {
    return found->second ;
  }
  switch (getOperatorType()) {
  case BINARY_OP :
    {
      if (leftChild == NULL || rightChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patFALSE;
      }

      patBoolean l = leftChild->isDerivativeStructurallyZero(index,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patFALSE ;
      }
      patBoolean r = rightChild->isDerivativeStructurallyZero(index,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patFALSE ;
      }

      derivStructurallyZero[index] = (l && r) ;
      return derivStructurallyZero[index] ;
    
    }    
  case UNARY_OP :
    {
      if (leftChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patFALSE;
      }

      patBoolean l = leftChild->isDerivativeStructurallyZero(index,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patFALSE;
      }
    
      derivStructurallyZero[index] = l ;
      return l ;
    }

  case VARIABLE_OP :
    {
      err = new patErrMiscError("This should have been overloaded") ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
  case RANDOM_OP :
    {
      err = new patErrMiscError("This should have been overloaded") ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
  case ATTRIBUTE_OP :
    {
      derivStructurallyZero[index] = patTRUE ;
      return patTRUE ;
    }

  case CONSTANT_OP :
    {
      derivStructurallyZero[index] = patTRUE ;
      return patTRUE ;
    }

  default:
    {
    err = new patErrMiscError("Operator not defined");
    WARNING(err->describe()) ;
    return patFALSE;
    }
  }

}
