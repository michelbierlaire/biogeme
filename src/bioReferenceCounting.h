//-*-c++-*------------------------------------------------------------
//
// File name : bioReferenceCounting.h
// @date   Wed Jun 17 12:11:35 2020
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioReferenceCounting_h
#define bioReferenceCounting_h

class bioReferenceCounting
{
    private:
    int count; // Reference count

    public:
    bioReferenceCounting() {
      count = 0 ;
    }
    void add() {
      count++;
    }

    int release() {
      return --count;
    }
};
#endif
