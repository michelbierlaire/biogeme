Native draws
============

The generation of draws is performed using the command ``bioDraws(’var’, ’DRAW_TYPE’)``, where the first argument, ``var``, 
provides the name of the random variable associated with the draws, and the second argument, ``DRAW_TYPE``, provides the 
distribution of the random variable (see the :class:`~biogeme.expressions.elementary_expressions.bioDraws` documentation). 
The draw type can either be a user-defined type 
(see `Bierlaire (2019) <http://transp-or.epfl.ch/documents/technicalReports/Bier19.pdf>`_ for details)
or native draws provided with Biogeme. The list if native doraws is provided in the following table.
    

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``UNIFORM``
     - Uniform U[0, 1]
   * - ``UNIFORM_ANTI``
     - Antithetic uniform U[0, 1]
   * - ``UNIFORM_HALTON2``
     - Halton draws with base 2, skipping the first 10
   * - ``UNIFORM_HALTON3``
     - Halton draws with base 3, skipping the first 10
   * - ``UNIFORM_HALTON5``
     - Halton draws with base 5, skipping the first 10
   * - ``UNIFORM_MLHS``
     - Modified Latin Hypercube Sampling on [0, 1]
   * - ``UNIFORM_MLHS_ANTI``
     - Antithetic Modified Latin Hypercube Sampling on [0, 1]
   * - ``UNIFORMSYM``
     - Uniform U[-1, 1]
   * - ``UNIFORMSYM_ANTI``
     - Antithetic uniform U[-1, 1]
   * - ``UNIFORMSYM_HALTON2``
     - Halton draws on [-1, 1] with base 2, skipping the first 10
   * - ``UNIFORMSYM_HALTON3``
     - Halton draws on [-1, 1] with base 3, skipping the first 10
   * - ``UNIFORMSYM_HALTON5``
     - Halton draws on [-1, 1] with base 5, skipping the first 10
   * - ``UNIFORMSYM_MLHS``
     - Modified Latin Hypercube Sampling on [-1, 1]
   * - ``UNIFORMSYM_MLHS_ANTI``
     - Antithetic Modified Latin Hypercube Sampling on [-1, 1]
   * - ``NORMAL``
     - Normal N(0, 1) draws
   * - ``NORMAL_ANTI``
     - Antithetic normal draws
   * - ``NORMAL_HALTON2``
     - Normal draws from Halton base 2 sequence
   * - ``NORMAL_HALTON3``
     - Normal draws from Halton base 3 sequence
   * - ``NORMAL_HALTON5``
     - Normal draws from Halton base 5 sequence
   * - ``NORMAL_MLHS``
     - Normal draws from Modified Latin Hypercube Sampling
   * - ``NORMAL_MLHS_ANTI``
     - Antithetic normal draws from Modified Latin Hypercube Sampling
.. toctree::
  :maxdepth: 2
  :caption: Native draws
