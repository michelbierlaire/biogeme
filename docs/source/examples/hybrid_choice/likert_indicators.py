from biogeme.latent_variables import LikertIndicator
from biogeme.latent_variables.likert_indicators import LikertType

likert_indicators = [
    LikertIndicator(
        name='Envir01',
        statement='Fuel price should be increased to reduce congestion and air pollution.',
        type='likert',
    ),
    LikertIndicator(
        name='Envir02',
        statement='More public transportation is needed, even if taxes are set to pay the additional costs.',
        type='likert',
    ),
    LikertIndicator(
        name='Envir03',
        statement='Ecology disadvantages minorities and small businesses.',
        type='likert',
    ),
    LikertIndicator(
        name='Envir04',
        statement='People and employment are more important than the environment.',
        type='likert',
    ),
    LikertIndicator(
        name='Envir05',
        statement='I am concerned about global warming.',
        type='likert',
    ),
    LikertIndicator(
        name='Envir06',
        statement='Actions and decision making are needed to limit greenhouse gas emissions.',
        type='likert',
    ),
    LikertIndicator(
        name='Mobil03',
        statement='I use the time of my trip in a productive way.',
        type='likert',
    ),
    LikertIndicator(
        name='Mobil05',
        statement='I reconsider frequently my mode choice.',
        type='likert',
    ),
    LikertIndicator(
        name='Mobil08',
        statement='I do not feel comfortable when I travel close to people I do not know.',
        type='likert',
    ),
    LikertIndicator(
        name='Mobil09',
        statement='Taking the bus helps making the city more comfortable and welcoming.',
        type='likert',
    ),
    LikertIndicator(
        name='Mobil10',
        statement='It is difficult to take the public transport when I travel with my children.',
        type='likert',
    ),
    LikertIndicator(
        name='Mobil12',
        statement='It is very important to have a beautiful car.',
        type='likert',
    ),
    LikertIndicator(
        name='LifSty01',
        statement='I always choose the best products regardless of price.',
        type='likert',
    ),
    LikertIndicator(
        name='LifSty07',
        statement='The pleasure of having something beautiful consists in showing it.',
        type='likert',
    ),
    LikertIndicator(
        name='NbCar',
        statement='Number of cars in the household',
        type='cars',
    ),
]

likert_types = [
    LikertType(
        type='likert',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[6, -1],
    ),
    LikertType(
        type='cars',
        symmetric=False,
        categories=[0, 1, 2, 3],
        neutral_labels=[-1],
        fix_first_cut_point_for_non_symmetric_thresholds=0.0,
    ),
]
