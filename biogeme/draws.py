""" Generation of various types of draws.

:author: Michel Bierlaire

:date: Tue Jun 18 19:05:13 2019
"""

# Too constraining
# pylint: disable=invalid-name, too-many-arguments, too-many-locals, too-many-statements

import numpy as np
import biogeme.exceptions as excep


def getUniform(sampleSize, numberOfDraws, symmetric=False):
    """Uniform [0, 1] or [-1, 1] numbers

    :param sampleSize: number of observations for which draws must be
                       generated. If None, a one dimensional array
                       will be generated. If it has a values k, then k
                       series of draws will be generated
    :type sampleSize: int
    :param numberOfDraws: number of draws to generate.
    :type numberOfDraws: int
    :param symmetric: if True, draws from [-1: 1] are generated.
        If False, draws from [0: 1] are generated.  Default: False
    :type symmetric: bool
    :return: numpy array with the draws
    :rtype: numpy.array

    Example::

        draws = dr.getUniform(sampleSize=3, numberOfDraws=10, symmetric=False)
        array(
            [[0.13053817, 0.63892308, 0.55031567, 0.26347854, 0.16730932,
              0.77745367, 0.48283887, 0.84247501, 0.20550219, 0.02373537],
             [0.68935846, 0.03363595, 0.36006669, 0.26709364, 0.54907706,
              0.22492104, 0.2494399 , 0.17323209, 0.52370401, 0.54091257],
             [0.40310204, 0.89916711, 0.86065005, 0.94277699, 0.09077065,
              0.40107731, 0.22554722, 0.47693135, 0.14058265, 0.17397031]]
        )

        draws = dr.getUniform(sampleSize=3, numberOfDraws=10, symmetric=True)
        array(
            [[ 0.74403237, -0.27995692,  0.33997421, -0.89405035, -0.129761  ,
               0.86593325,  0.30657422,  0.82435619,  0.498482  ,  0.24561616],
             [-0.48239607, -0.29257815, -0.98342034,  0.68392813, -0.25379429,
               0.49359859, -0.26459883,  0.14569724, -0.68860467, -0.40903446],
             [ 0.93251627, -0.85166912,  0.58096917,  0.39289882, -0.65088635,
               0.40114744, -0.61327161,  0.08900539, -0.20985417,  0.67542226]]
        )

    :raise biogemeError: if the number of draws is not positive.

    :raise biogemeError: if the sample size is not positive.
    """
    if numberOfDraws <= 0:
        raise excep.biogemeError(f'Invalid number of draws: {numberOfDraws}.')

    if sampleSize <= 0:
        raise excep.biogemeError(
            f'Invalid sample size: {sampleSize} when generating draws.'
        )
    totalSize = numberOfDraws * sampleSize

    uniformNumbers = np.random.uniform(size=totalSize)
    if symmetric:
        uniformNumbers = 2.0 * uniformNumbers - 1.0

    uniformNumbers.shape = (sampleSize, numberOfDraws)
    return uniformNumbers


def getLatinHypercubeDraws(
    sampleSize, numberOfDraws, symmetric=False, uniformNumbers=None
):
    """Implementation of the Modified Latin Hypercube Sampling proposed
    by `Hess et al., (2006)`_.

    .. _`Hess et al., (2006)`:
       https://doi.org/10.1016/j.trb.2004.10.005

    :param sampleSize: number of observations for which draws must be
                       generated. If None, a one dimensional array
                       will be generated. If it has a values k, then k
                       series of draws will be generated
    :type sampleSize: int
    :param numberOfDraws: number of draws to generate.
    :type numberOfDraws: int
    :param symmetric: if True, draws from [-1: 1] are generated.
       If False, draws from [0: 1] are generated.  Default: False
    :type symmetric: bool
    :param uniformNumbers: numpy with uniformly distributed numbers.
       If None, the numpy uniform number generator is used.
    :type uniformNumbers: numpy.array

    :return: numpy array with the draws
    :rtype: numpy.array

    Example::

        latinHypercube = dr.getLatinHypercubeDraws(sampleSize=3,
                                                   numberOfDraws=10)
        array([[0.43362897, 0.5275741 , 0.09215663, 0.94056236, 0.34376868,
                0.87195551, 0.41495219, 0.71736691, 0.23198736, 0.145561  ],
               [0.30520544, 0.78082964, 0.83591146, 0.2733167 , 0.53890906,
                0.61607469, 0.00699715, 0.17179441, 0.7557228 , 0.39733102],
               [0.49676864, 0.67073483, 0.9788854 , 0.5726069 , 0.11894558,
                0.05515471, 0.2640275 , 0.82093696, 0.92034628, 0.64866597]])

    :raise biogemeError: if the number of draws is not positive.

    :raise biogemeError: if the sample size is not positive.

    :raise biogemeError: if the number of uniform draws is inconsistent.
    """
    if numberOfDraws <= 0:
        raise excep.biogemeError(f'Invalid number of draws: {numberOfDraws}.')

    if sampleSize <= 0:
        raise excep.biogemeError(
            f'Invalid sample size: {sampleSize} when generating draws.'
        )
    totalSize = numberOfDraws * sampleSize

    if uniformNumbers is None:
        uniformNumbers = np.random.uniform(size=totalSize)
    else:
        if uniformNumbers.size != totalSize:
            errorMsg = (
                f'A total of {totalSize} uniform draws '
                f'must be provided, and not {uniformNumbers.size}.'
            )
            raise excep.biogemeError(errorMsg)

    uniformNumbers.shape = (totalSize,)
    numbers = np.array(
        [
            (float(i) + uniformNumbers[i]) / float(totalSize)
            for i in range(totalSize)
        ]
    )
    if symmetric:
        numbers = 2.0 * numbers - 1.0

    np.random.shuffle(numbers)
    numbers.shape = (sampleSize, numberOfDraws)
    return numbers


def getHaltonDraws(
    sampleSize, numberOfDraws, symmetric=False, base=2, skip=0, shuffled=False
):
    """Generate Halton draws.
    Implementation by Cristian Arteaga, University of Nevada Las Vegas,

    :param sampleSize: number of observations for which draws must be
                       generated. If None, a one dimensional array
                       will be generated. If it has a values k, then k
                       series of draws will be generated
    :type sampleSize: int

    :param numberOfDraws: number of draws to generate.
    :type numberOfDraws: int

    :param symmetric: if True, draws from [-1: 1] are generated.
           If False, draws from [0: 1] are generated.  Default: False
    :type symmetric: bool

    :param base: generate Halton draws for a given basis.
            Ideally, it should be a prime number. Default: 2.
    :type base: int

    :param skip: the number of  elements of the sequence to be discarded.
    :type skip: int

    :param shuffled: if True, each series is shuffled
    :type shuffled: bool

    :return: numpy array with the draws
    :rtype: numpy.array

    Example::

        halton = dr.getHaltonDraws(sampleSize=2, numberOfDraws=10, base=3)
        array([[0.33333333, 0.66666667, 0.11111111, 0.44444444, 0.77777778,
                0.22222222, 0.55555556, 0.88888889, 0.03703704, 0.37037037],
               [0.7037037 , 0.14814815, 0.48148148, 0.81481481, 0.25925926,
                0.59259259, 0.92592593, 0.07407407, 0.40740741, 0.74074074]])

    :raise biogemeError: if the number of draws is not positive.

    :raise biogemeError: if the sample size is not positive.
    """
    if numberOfDraws <= 0:
        raise excep.biogemeError(f'Invalid number of draws: {numberOfDraws}.')

    if sampleSize <= 0:
        raise excep.biogemeError(
            f'Invalid sample size: {sampleSize} when generating draws.'
        )
    length = numberOfDraws * sampleSize
    req_length = length + skip + 1
    numbers = np.empty(req_length)
    numbers[0] = 0
    numbers_idx = 1
    t = 1
    while numbers_idx < req_length:
        d = 1 / base**t
        numbers_size = numbers_idx
        i = 1
        while i < base and numbers_idx < req_length:
            max_numbers = min(req_length - numbers_idx, numbers_size)
            numbers[numbers_idx : numbers_idx + max_numbers] = (
                numbers[:max_numbers] + d * i
            )
            numbers_idx += max_numbers
            i += 1
        t += 1
    numbers = numbers[skip + 1 : length + skip + 1]

    if shuffled:
        np.random.shuffle(numbers)

    if symmetric:
        numbers = 2.0 * numbers - 1.0

    numbers.shape = (sampleSize, numberOfDraws)
    return numbers


def getAntithetic(unif, sampleSize, numberOfDraws):
    """Returns antithetic uniform draws

    :param unif: function taking two arguments (sampleSize, numberOfDraws)
                 and returning U[0, 1] draws
    :type unif: function

    :param sampleSize: number of observations for which draws must be
                       generated. If None, a one dimensional array
                       will be generated. If it has a values k, then k
                       series of draws will be generated
    :type sampleSize: int

    :param numberOfDraws: number of draws to generate.
    :type numberOfDraws: int

    :return: numpy array with the antithetic draws
    :rtype: numpy.array

    Example::

          draws = dr.getAntithetic(dr.getUniform,
                                   sampleSize=3,
                                   numberOfDraws=10)

          array([[0.48592363, 0.13648133, 0.35925946, 0.32431338, 0.32997936,
                  0.51407637, 0.86351867, 0.64074054, 0.67568662, 0.67002064],
                 [0.89261997, 0.0331808 , 0.30767182, 0.93433648, 0.17196124,
                  0.10738003, 0.9668192 , 0.69232818, 0.06566352, 0.82803876],
                 [0.81095587, 0.96171364, 0.40984817, 0.72177258, 0.16481096,
                  0.18904413, 0.03828636, 0.59015183, 0.27822742, 0.83518904]])

    """
    R = int(numberOfDraws / 2)
    draws = unif(sampleSize, R)
    return np.concatenate((draws, 1 - draws), axis=1)


def getNormalWichuraDraws(
    sampleSize, numberOfDraws, uniformNumbers=None, antithetic=False
):
    """Generate pseudo-random numbers from a normal distribution N(0, 1)

    It uses the Algorithm AS241 by `Wichura (1988)`_
    which produces the normal deviate z corresponding to a given lower
    tail area of p; z is accurate to about 1 part in :math:`10^{16}`.

    .. _`Wichura (1988)`:
       http://www.jstor.org/stable/2347330

    :param sampleSize: number of observations for which draws must be
                       generated. If None, a one dimensional array
                       will be generated. If it has a values k, then k
                       series of draws will be generated
    :type sampleSize: int

    :param numberOfDraws: number of draws to generate.
    :type numberOfDraws: int

    :param uniformNumbers: numpy with uniformly distributed numbers.
               If None, the numpy uniform number generator is used.
    :type uniformNumbers: numpy.array
    :param antithetic: if True, only half of the draws are
                       actually generated, and the series are completed
                       with their antithetic version.
    :type antithetic: bool

    :return: numpy array with the draws
    :rtype: numpy.array

    Example::

        draws = dr.getNormalWichuraDraws(sampleSize=3, numberOfDraws=10)
        array(
            [[ 0.52418458, -1.04344204, -2.11642482,  0.48257162, -2.67188279,
              -1.89993283,  0.28251041, -0.38424425,  1.53182226,  0.30651874],
             [-0.7937038 , -0.07884121, -0.91005616, -0.98855175,  1.09405753,
              -0.5997651 , -1.70785113,  1.57571384, -0.33208723, -1.03510102],
             [-0.13853654,  0.92595498, -0.80136586,  1.68454196,  0.9955927 ,
              -0.28615154,  2.10635541,  0.0436191 , -0.25417774,  0.01026933]]
        )

    :raise biogemeError: if the number of draws is not positive.

    :raise biogemeError: if the sample size is not positive.

    """
    if numberOfDraws <= 0:
        raise excep.biogemeError(f'Invalid number of draws: {numberOfDraws}.')

    if antithetic:
        if 2 * int(numberOfDraws / 2) != numberOfDraws:
            errorMsg = (
                f'Please specify an even number of draws for '
                f'antithetic draws. Requested number of '
                f'{numberOfDraws}.'
            )
            raise excep.biogemeError(errorMsg)
        numberOfDraws = int(numberOfDraws / 2)

    if sampleSize <= 0:
        raise excep.biogemeError(
            f'Invalid sample size: {sampleSize} when generating draws.'
        )
    totalSize = numberOfDraws * sampleSize

    split2 = 5.0e00
    const1 = 0.180625e00
    const2 = 1.6e00
    a0 = 3.3871328727963666080e00
    a1 = 1.3314166789178437745e02
    a2 = 1.9715909503065514427e03
    a3 = 1.3731693765509461125e04
    a4 = 4.5921953931549871457e04
    a5 = 6.7265770927008700853e04
    a6 = 3.3430575583588128105e04
    a7 = 2.5090809287301226727e03
    b1 = 4.2313330701600911252e01
    b2 = 6.8718700749205790830e02
    b3 = 5.3941960214247511077e03
    b4 = 2.1213794301586595867e04
    b5 = 3.9307895800092710610e04
    b6 = 2.8729085735721942674e04
    b7 = 5.2264952788528545610e03
    c0 = 1.42343711074968357734e00
    c1 = 4.63033784615654529590e00
    c2 = 5.76949722146069140550e00
    c3 = 3.64784832476320460504e00
    c4 = 1.27045825245236838258e00
    c5 = 2.41780725177450611770e-01
    c6 = 2.27238449892691845833e-02
    c7 = 7.74545014278341407640e-04
    d1 = 2.05319162663775882187e00
    d2 = 1.67638483018380384940e00
    d3 = 6.89767334985100004550e-01
    d4 = 1.48103976427480074590e-01
    d5 = 1.51986665636164571966e-02
    d6 = 5.47593808499534494600e-04
    d7 = 1.05075007164441684324e-09
    e0 = 6.65790464350110377720e00
    e1 = 5.46378491116411436990e00
    e2 = 1.78482653991729133580e00
    e3 = 2.96560571828504891230e-01
    e4 = 2.65321895265761230930e-02
    e5 = 1.24266094738807843860e-03
    e6 = 2.71155556874348757815e-05
    e7 = 2.01033439929228813265e-07
    f1 = 5.99832206555887937690e-01
    f2 = 1.36929880922735805310e-01
    f3 = 1.48753612908506148525e-02
    f4 = 7.86869131145613259100e-04
    f5 = 1.84631831751005468180e-05
    f6 = 1.42151175831644588870e-07
    f7 = 2.04426310338993978564e-15

    if uniformNumbers is None:
        uniformNumbers = np.random.uniform(size=totalSize)
    elif uniformNumbers.size != totalSize:
        errorMsg = (
            f'A total of {totalSize} uniform draws must be '
            f'provided, and not {uniformNumbers.size}.'
        )
        raise excep.biogemeError(errorMsg)
    uniformNumbers.shape = (totalSize,)

    q = uniformNumbers - 0.5
    draws = np.zeros(uniformNumbers.shape)
    r = np.zeros(uniformNumbers.shape)
    cond1 = np.abs(uniformNumbers) <= 0.45
    r[cond1] = const1 - q[cond1] * q[cond1]
    draws[cond1] = (
        q[cond1]
        * (
            (
                (
                    (
                        (
                            ((a7 * r[cond1] + a6) * r[cond1] + a5) * r[cond1]
                            + a4
                        )
                        * r[cond1]
                        + a3
                    )
                    * r[cond1]
                    + a2
                )
                * r[cond1]
                + a1
            )
            * r[cond1]
            + a0
        )
        / (
            (
                (
                    (
                        (
                            ((b7 * r[cond1] + b6) * r[cond1] + b5) * r[cond1]
                            + b4
                        )
                        * r[cond1]
                        + b3
                    )
                    * r[cond1]
                    + b2
                )
                * r[cond1]
                + b1
            )
            * r[cond1]
            + 1
        )
    )
    cond2 = np.abs(uniformNumbers) > 0.45
    cond2a = np.logical_and(cond2, q < 0.0)
    cond2b = np.logical_and(cond2, q >= 0.0)
    r[cond2a] = uniformNumbers[cond2a]
    r[cond2b] = 1 - uniformNumbers[cond2b]
    cond2c = np.logical_and(cond2, r <= 0)
    cond2d = np.logical_and(cond2, r > 0)
    draws[cond2c] = 0.0
    r[cond2d] = np.sqrt(-np.log(r[cond2d]))
    cond2d_a = np.logical_and(cond2d, r <= split2)
    cond2d_b = np.logical_and(cond2d, r > split2)
    r[cond2d_a] = r[cond2d_a] - const2
    draws[cond2d_a] = (
        (
            (
                (
                    (
                        ((c7 * r[cond2d_a] + c6) * r[cond2d_a] + c5)
                        * r[cond2d_a]
                        + c4
                    )
                    * r[cond2d_a]
                    + c3
                )
                * r[cond2d_a]
                + c2
            )
            * r[cond2d_a]
            + c1
        )
        * r[cond2d_a]
        + c0
    ) / (
        (
            (
                (
                    (
                        ((d7 * r[cond2d_a] + d6) * r[cond2d_a] + d5)
                        * r[cond2d_a]
                        + d4
                    )
                    * r[cond2d_a]
                    + d3
                )
                * r[cond2d_a]
                + d2
            )
            * r[cond2d_a]
            + d1
        )
        * r[cond2d_a]
        + 1
    )
    r[cond2d_b] = r[cond2d_b] - split2
    draws[cond2d_b] = (
        (
            (
                (
                    (
                        ((e7 * r[cond2d_b] + e6) * r[cond2d_b] + e5)
                        * r[cond2d_b]
                        + e4
                    )
                    * r[cond2d_b]
                    + e3
                )
                * r[cond2d_b]
                + e2
            )
            * r[cond2d_b]
            + e1
        )
        * r[cond2d_b]
        + e0
    ) / (
        (
            (
                (
                    (
                        ((f7 * r[cond2d_b] + f6) * r[cond2d_b] + f5)
                        * r[cond2d_b]
                        + f4
                    )
                    * r[cond2d_b]
                    + f3
                )
                * r[cond2d_b]
                + f2
            )
            * r[cond2d_b]
            + f1
        )
        * r[cond2d_b]
        + 1
    )
    draws[cond2a] = -draws[cond2a]

    draws.shape = (sampleSize, numberOfDraws)

    if antithetic:
        draws = np.concatenate((draws, -draws), axis=1)

    return draws
