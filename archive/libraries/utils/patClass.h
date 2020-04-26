// BaseClass.h
//
// Routine manager: "Burton"                               Date: 26-05-95
//
// Routine history (creation,modification,correction)
// +--------------------------------------------------------------------+
// !Programmer ! Comments                              ! Date   !Version!
// +-----------!---------------------------------------!--------!-------+
// ! Burton    ! Creation                              !26-05-95! 1.0-00!
// +--------------------------------------------------------------------+
//----------------------------------------------------------------------+
//                                                                      !
// - CLASS NAME  :                                                      !
//                                                                      !
// - FILE        : [X] Interface      (.h)                              !
//                 [X] Implementation (.cc)                             !
//                                                                      !
// - CLASS TYPE  : [ ] Abstract      [ ] Representation                 !
//                 [ ] Base          [ ] Derived                        !
//                 [ ] Template      [X] Macros                         !
//                                                                      !
// - DERIVATION  : [ ] Public    : .................................... !
//                 [ ] Protected : .................................... !
//                 [ ] Private   : .................................... !
//                 [ ] Virtual   : [ ] All                              !
//                                 [ ] : .............................. !
//                                                                      !
// - LAYERED     : [ ] with class : ................................... !
//                                                                      !
// - FRIENDS     :                                                      !
//                                                                      !
// - EXTERNALS   :                                                      !
//                                                                      !
// - C++         : [X] 2.0                                              !
//                 [ ] 2.1  (embedded types)                            !
//                 [X] 3.0  (templates & exceptions)                    !
//                                                                      !
// - CALLS TO    : [ ] I/O streams    [ ] X11                           !
//                 [ ] strings        [ ] X11 Extensions                !
//                 [ ] math           [ ] X11 Toolkit                   !
//                 [ ] system         [ ] OSF/MOTIF (Xm library)        !
//                 [ ] GNU library    [ ] Xmt - Xpm - other X libraries !
//                                                                      !
// - DESCRIPTION : Provide macros for automatically declaring and       !
//                 defining dtor, copy ctor and operator= of base       !
//                 classes. Separate macros are given for templates.    !
//                                                                      !
//----------------------------------------------------------------------+

#ifndef patClass_h
#define patClass_h
                                    // includes from C++ library

  // In these macros, inlined functions SHOULD not be virtual, because
  // c++ compilers cannot effectively inline virtual fcts. Then they
  // are virtual and not inlined. For the sake of efficiency, we
  // deliberately omit the use of this keyword in the macros (except for
  // destructors (of course), cleanup(), and transfer().

                                    // give common member functions: cleanup()
                                    // which is used in the destructor,
                                    // transfer(.) which is used for the copy
                                    // ctor and the operator=.
                                    // Use this in .h file.

#define patClass_declare(CLASS)          \
protected:                               \
  void cleanup();                        \
  void transfer( const CLASS& );         \
public:                                  \
  virtual ~CLASS();                      \
  CLASS( const CLASS& );                 \
  CLASS& operator=( const CLASS& );

                                    // Implement the dtor, copy ctor, and
                                    // operator=.
                                    // Use this in .cc file.

#define patClass_define(CLASS)          \
                                         \
CLASS::~CLASS() {                        \
  cleanup();                             \
}                                        \
                                         \
CLASS::CLASS( const CLASS& obj ) {       \
  transfer( obj );                       \
}                                        \
                                         \
CLASS&                                   \
CLASS::operator=( const CLASS& obj ) {   \
  if ( this != &obj ) {                  \
    cleanup();                           \
    transfer( obj );                     \
  }                                      \
  return (*this);                        \
}

                                    // Idem for the implementation of template
                                    // classes.

#define patClass_defineTemplate(CLASS)      \
                                             \
template <class T> CLASS<T>::~CLASS() {      \
  cleanup();                                 \
}                                            \
                                             \
template <class T>                           \
CLASS<T>::CLASS( const CLASS<T>& obj ) {     \
  transfer( obj );                           \
}                                            \
                                             \
template <class T> CLASS<T>&                 \
CLASS<T>::operator=( const CLASS<T>& obj ) { \
  if ( this != &obj ) {                      \
    cleanup();                               \
    transfer( obj );                         \
  }                                          \
  return (*this);                            \
}

                                    // define a protected member data and its
                                    // public read access to it.

#define patClass_data_r(TYPE,DATA)           \
                                              \
protected:                                    \
  TYPE name2(_,DATA);                         \
public:                                       \
  TYPE DATA() const { return name2(_,DATA); }

                                    // define a protected member data as a ptr
                                    // and its public read access to it.

#define patClass_dataPtr_r(TYPE,DATA)         \
                                               \
protected:                                     \
  TYPE *name2(_,DATA);                         \
public:                                        \
  TYPE *DATA() const { return name2(_,DATA); }

                                    // define a protected member data, its
                                    // public read access to it, and a function
                                    // to modify its value from a data of same
                                    // type. The set() fct uses operator= of
                                    // the considered type.

#define patClass_data_w(TYPE,DATA)           \
                                              \
protected:                                    \
  TYPE name2(_,DATA);                         \
public:                                       \
  void name2(DATA,Set)( TYPE d_ ) {           \
         name2(_,DATA) = d_; }                \
  TYPE DATA() const { return name2(_,DATA); }

                                    // in next version set() takes its argument
                                    // by reference.

#define patClass_rdata_w(TYPE,DATA)          \
                                              \
protected:                                    \
  TYPE name2(_,DATA);                         \
public:                                       \
  void name2(DATA,Set)( TYPE & d_ ) {         \
         name2(_,DATA) = d_; }                \
  TYPE DATA() const { return name2(_,DATA); }

                                    // idem with pointer data.

#define patClass_dataPtr_w(TYPE,DATA)         \
                                               \
protected:                                     \
  TYPE *name2(_,DATA);                         \
public:                                        \
  void name2(DATA,Set)( TYPE * d_ ) {          \
         name2(_,DATA) = d_; }                 \
  TYPE *DATA() const { return name2(_,DATA); }

                                    // idem with pointer data and deep copy

#define patClass_dataPtr_dw(TYPE,DATA)        \
                                               \
protected:                                     \
  TYPE *name2(_,DATA);                         \
public:                                        \
  void name2(DATA,Set)( TYPE * d_ ) {          \
         name2(_,DATA) = d_; }                 \
  int name2(DATA,SetDeep)( TYPE * d_ );        \
  TYPE *DATA() const { return name2(_,DATA); }

                                    // define the setDeep() function. We do not
                                    // inline this function to avoid #include
                                    // "TYPE.h" in the .h class file.

#define patClass_defineDeepCopyPtr(HOST,TYPE,DATA)  \
                                                     \
int HOST::name2(DATA,SetDeep)( TYPE * d_ ) {         \
  if ( name2(_,DATA) && d_ ) {                       \
    (*name2(_,DATA)) = (*d_);                        \
    return 1;                                        \
  }                                                  \
  else if ( !d_ ) {                                  \
    delete name2(_,DATA);                            \
    name2(DATA,Set)( d_ );                           \
    return 1;                                        \
  }                                                  \
  return 0;                                          \
}

#endif // patClass_h
