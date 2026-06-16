"""
Likert indicators and Likert types (SPEC ONLY)

Pure semantic indicator declarations
Michel Bierlaire
Sat Mar 14 2026, 16:26:11
"""

from biogeme.latent_variables import LikertIndicator, LikertType

likert_indicators = [
    LikertIndicator(
        name="Envir01",
        statement="Fuel price should be increased to reduce congestion and air pollution.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Envir02",
        statement="More public transportation is needed, even if taxes are set to pay the additional costs.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Envir03",
        statement="Ecology disadvantages minorities and small businesses.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Envir04",
        statement="People and employment are more important than the environment.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Envir05",
        statement="I am concerned about global warming.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Envir06",
        statement="Actions and decision making are needed to limit greenhouse gas emissions.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Mobil03",
        statement="I use the time of my trip in a productive way.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Mobil05",
        statement="I reconsider frequently my mode choice.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Mobil08",
        statement="I do not feel comfortable when I travel close to people I do not know.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Mobil09",
        statement="Taking the bus helps making the city more comfortable and welcoming.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Mobil10",
        statement="It is difficult to take the public transport when I travel with my children.",
        type_name="likert",
    ),
    LikertIndicator(
        name="Mobil12",
        statement="It is very important to have a beautiful car.",
        type_name="likert",
    ),
    LikertIndicator(
        name="LifSty01",
        statement="I always choose the best products regardless of price.",
        type_name="likert",
    ),
    LikertIndicator(
        name="LifSty07",
        statement="The pleasure of having something beautiful consists in showing it.",
        type_name="likert",
    ),
]

likert_types = [
    LikertType(
        type_name="likert",
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[6, -1],
    ),
]
