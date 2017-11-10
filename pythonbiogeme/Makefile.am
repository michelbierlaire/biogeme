ACLOCAL_AMFLAGS = -I m4
pkgdatdir = ${pkgdatadir}$(INSTALL_VERSION_SUFFIX)

if BUILD_PYTHON
bin_SCRIPTS = pythonbiogeme
if BUILD_STATIC
bin_SCRIPTS += pythonbiogeme_static
endif
if BUILD_DEBUG
bin_SCRIPTS += debugbiogeme
endif
if INSTALL_VERSIONED
install-exec-hook:
	cd $(DESTDIR)$(bindir) && \
	for p in $(bin_SCRIPTS); do \
	mv -f $${p}$(EXEEXT) $${p}$(INSTALL_VERSION_SUFFIX)$(EXEEXT) && \
	$(LN_S) $${p}$(INSTALL_VERSION_SUFFIX)$(EXEEXT) $${p}$(EXEEXT); \
	done
endif
endif

CLEANFILES = $(bin_SCRIPTS)

dist_pkgdat_DATA = \
python/bio_expression.py \
python/bio_iterator.py \
python/biogeme.py \
python/loglikelihood.py \
python/statistics.py \
python/simulateDiscreteDistribution.py \
python/distributions.py \
python/nested.py \
python/cnl.py \
python/mev.py \
python/boxcox.py \
python/legendre.py \
python/bioMatrix.py \
python/weightedloglikelihood.py 

#EXTRA_DIST = python/biogeme-python3.1-config.py python/biogeme-python3.2-config.py python/biogeme-python3.3-config.py python/biogeme-python3.4-config.py  python/biogeme-python3.5-config.py

EXTRA_DIST = python/biogeme-python3-config.py 
