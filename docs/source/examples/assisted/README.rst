Assisted specification with Biogeme
***********************************

Examples discussed in `Bierlaire and Ortelli (2023) Assisted Specification with Biogeme 3.2.12
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_

The example :ref:`plot_b10_parameter_overrides` reproduces a common missing
category problem.  A ``has_pt_subscr`` segmentation contains one ``-99``
observation, represented in the segmentation as ``minus_99``, and the
automatically generated coefficients for that category are fixed to zero in
every catalog alternative before estimation.
