"""Sample datasets for demonstrating and testing agreement coefficients."""

import pandas as pd


def dist_4cat():
    """Distribution of 4 raters by Category and Subject - Subjects allocated \
    in 2 groups A and B.

    This dataset summarizes the ratings assigned by 4 raters who classified 15
    subjects into one of 3 categories named "a", "b", and "c".

    Returns
    -------
    pandas.DataFrame

        Group
            This variable represents the group name.
        a
            Number of ratings in category "a"
        b
            Number of ratings in category "b"
        c
            Number of ratings in category "c"
    """
    group = pd.Index("A A A A A A A A A B B B B B B".split(), name="Group")
    data = dict(
        a=[3, 2, 2, 2, 3, 3, 0, 0, 0, 3, 0, 0, 0, 0, 3],
        b=[0, 1, 1, 0, 1, 1, 4, 3, 0, 1, 4, 3, 2, 0, 1],
        c=[1, 1, 1, 2, 0, 0, 0, 1, 4, 0, 0, 1, 2, 4, 0],
    )
    return pd.DataFrame(data, group)


def dist_g1g2():
    """Distribution of 4 raters by subject and by category, for 14 Subjects \
    that belong to 2 groups "G1" and "G2".

    This dataset contains rating data in the form of a subject-level
    distribution of 4 raters by category the subject was classified into.
    A total of 4 raters had to classify 14 subjects into one of 5 categories
    "a", "b", "c", "d", and "e". None of the 4 raters scored all 14 units.
    Therefore, some missing ratings appear in each of the columns associated
    with the 4 raters.

    This dataset is different version of the more detailed `raw_g1g2` dataset.
    While `raw_g1g2` tells you about the exact category into which each rater
    classified all subjects, this one on the other hand, can only tell you how
    many raters classified a given subject into a particular category.

    See Also
    --------
    :meth:`~irrCAC.datasets.raw_g1g2()`

    Returns
    -------
    pandas.DataFrame

        Group
            This variable represents the group name.
        Units
            This variable represents the unit number.
        a
            Number of raters who classified the subject represented by the row
            into category "a"
        b
            Number of raters who classified the subject represented by the row
            into category "b"
        c
            Number of raters who classified the subject represented by the row
            into category "c"
        d
            Number of raters who classified the subject represented by the row
            into category "d"
        e
            Number of raters who classified the subject represented by the row
            into category "e"
    """
    group = "G2 G1 G2 G1 G1 G1 G2 G1 G2 G2 G1 G1 G2 G2".split()
    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    index = pd.MultiIndex.from_arrays([group, units], names=("Group", "Units"))
    data = dict(
        a=[3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2, 0, 0, 1],
        b=[0, 3, 0, 0, 4, 1, 0, 1, 4, 0, 0, 0, 3, 2],
        c=[0, 1, 4, 4, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        d=[0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0],
        e=[0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    )
    return pd.DataFrame(data, index=index)


def raw_4raters():
    """Rating Data from 4 Raters and 12 Subjects.

    This dataset :cite:p:`Gwe14` contains ratings obtained from an
    experiment where 4 raters classified 12 subjects into 5 possible categories labeled
    as 1, 2, 3, 4, and 5. None of the 4 raters scored all 12 units. Therefore, some
    missing ratings in the form of "NA" appear in each of the columns associated with
    the 4 raters. Note that only the 4 last columns are to be used with the functions
    included in this package. The first column only plays a descriptive role and is not
    used in any calculation.

    Returns
    -------
    pandas.DataFrame
        The data frame has the following columns:

        Units
            This variable represents the unit number.
        Rater1
            All ratings from rater 1
        Rater2
            All ratings from rater 2
        Rater3
            All ratings from rater 3
        Rater4
            All ratings from rater 4
    """
    units = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], name="Units")
    data = dict(
        Rater1=[1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        Rater2=[1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, None],
        Rater3=[None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, 3],
        Rater4=[1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    )

    return pd.DataFrame(data, index=units)


def raw_5observers():
    """Scores assigned by 5 observers to 20 experimental units.

    This dataset contains data from a reliability experiment where 5 observers
    scored 15 units on a 4-point numeric scale based on the values
    0, 1, 2 and 3.

    The dataset :cite:p:`Gwe14` has 15 rows (for the 15 subjects) and 6
    columns. Only the last 5 columns associated with the 5 observers are used in the
    calculations. Of the 5 observers, only observer 3 scored all 15 units.
    Therefore, some missing ratings in the form of "NA" appear in the columns
    associated with the remaining 4 observers.

    Returns
    -------
    pandas.DataFrame
        The data frame has the following columns:

        Units
            This variable represents the unit number.
        Observer1
            All ratings from observer 1
        Observer2
            All ratings from observer 2
        Observer3
            All ratings from observer 3
        Observer4
            All ratings from observer 4
        Observer5
            All ratings from observer 5

    """
    units = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], name="Units")
    data = dict(
        Observer1=[1, 1, 2, None, 0, 0, 1, 1, 2, 2, None, 0, 1, 3, 1],
        Observer2=[1, 1, 3, 0, 0, 0, 0, None, 2, 1, 1, 0, 2, 3, 1],
        Observer3=[2, 0, 3, 0, 0, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1],
        Observer4=[None, 1, 3, None, None, None, None, 0, None, 1, 0, 0, 2, 2, None],
        Observer5=[2, None, None, 0, 0, 0, 1, None, 2, None, None, None, None, 3, 1],
    )
    return pd.DataFrame(data, index=units, dtype=float)


def raw_ben_gerry():
    """Ratings of 12 units from 2 raters named Ben and Gerry.

    This dataset contains ratings that 2 raters named Ben and Gerry assigned to
    12 units distributed in 2 groups "G1" and "G2". Each row of this dataset
    describes an interval and the interpretation of the magnitude it represents.

    Returns
    -------
    pandas.Dataframe
        The data frame has the following columns:

        Group
            This variable represents the group membership.
        Units
            This variable represents the unit number.
        Ben
            All ratings from Ben
        Gerry
            All ratings from Gerry
    """
    group = "G2 G2 G1 G1 G1 G2 G1 G2 G2 G1 G1 G2".split()
    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    index = pd.MultiIndex.from_arrays([group, units], names=("Group", "Units"))
    data = dict(
        Ben=["a", "a", "b", "b", "d", "c", "c", "c", "e", "d", "d", "a"],
        Gerry=["b", "a", "b", "b", "b", "c", "c", "c", "e", None, "d", "d"],
    )
    return pd.DataFrame(data, index=index)


def raw_g1g2():
    """Dataset of raw ratings from 4 Raters on 14 Subjects that belong to 2 \
    groups "G1" and "G2".

    This dataset contains data from a reliability experiment where 4 raters
    identified as Rater1, Rater2, Rater3 and Rater4 scored 14 units on a
    5-point alphabetical scale based on the values a, b, c, d and e.
    These 14 units are allocated to 2 groups named G1 and G2

    This dataset contains ratings obtained from an experiment where 4 raters
    classified 14 subjects into 5 possible categories labeled as a, b, c, d,
    and e. None of the 4 raters scored all 14 units. Therefore, some missing
    ratings appear in each of the columns associated with the 4 raters.

    Returns
    -------
    pandas.DataFrame
        The data frame has the following columns:

        Group
            This variable represents the group membership.
        Units
            This variable represents the unit number.
        Rater1
            All ratings from rater 1
        Rater2
            All ratings from rater 2
        Rater3
            All ratings from rater 3
        Rater4
            All ratings from rater 4
    """
    group = "G2 G1 G2 G1 G1 G1 G2 G1 G2 G2 G1 G1 G2 G2".split()
    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    index = pd.MultiIndex.from_arrays([group, units], names=("Group", "Units"))
    data = dict(
        Rater1=[
            "a",
            "b",
            "c",
            "c",
            "b",
            "a",
            "d",
            "a",
            "b",
            None,
            None,
            None,
            "b",
            "b",
        ],
        Rater2=["a", "b", "c", "c", "b", "b", "d", "a", "b", "e", None, None, "b", "a"],
        Rater3=[None, "c", "c", "c", "b", "c", "d", "b", "b", "e", "a", "c", "c", "c"],
        Rater4=["a", "b", "c", "c", "b", "d", "d", "a", "b", "e", "a", None, "b", "b"],
    )
    return pd.DataFrame(data, index=index)


def raw_gender():
    """ Rating Data from 4 Raters and 15 human Subjects, 9 of whom are female \
    and 6 males.

    This dataset contains data from a reliability experiment where 4 raters
    scored 15 units on a 3-point alphabetic scale based on the values
    a, b, and c.

    Returns
    -------
    pandas.DataFrame
        The data frame has the following columns:

        Group
            This variable represents the group membership.
        Units
            This variable represents the unit number.
        Rater1
            All ratings from rater 1
        Rater2
            All ratings from rater 2
        Rater3
            All ratings from rater 3
        Rater4
            All ratings from rater 4
    """
    group = "M M M M M M M M M F F F F F F".split()

    units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    index = pd.MultiIndex.from_arrays([group, units], names=("Group", "Units"))
    data = dict(
        Rater1="a a a a a b b b c a a a a a b".split(),
        Rater2="a a a a b a b c c a a a a b a".split(),
        Rater3="a b b c a a b b c a b b c a a".split(),
        Rater4="c c c c a a b b c c c c c a a".split(),
    )
    return pd.DataFrame(data, index=index)


def table_cont3x3abstractors():
    """Distribution of 100 pregnant women by pregnancy type and by abstractor.

    This dataset :cite:p:`Gwe14` contains pregnancy type data collected from 100 women
    who entered an Emergency Room with a positive pregnancy test and a second condition,
    which is either abdominal pain or vaginal bleeding. After reviewing their medical
    records, 2 reviewers (also referred to as abstractors) classified them into one of
    the following three pregnancy categories: Ectopic Pregnancy (Ectopic),
    Abnormal Intrauterine pregnancy (AIU) and Normal Intrauterine Pregnancy (NIU).

    Each row of this dataset describes an interval and the interpretation of the
    magnitude it represents.

    Returns
    -------
    pandas.DataFrame

        Ectopic
            Ectopic Pregnancy
        AIU
            Abnormal Intrauterine Pregnancy
        NIU
            Normal Intrauterine Pregnancy

    """
    data = {"Ectopic": [13, 0, 0], "AIU": [0, 20, 4], "NIU": [0, 7, 56]}
    return pd.DataFrame(data, index=list(data.keys()))


def table_cont4x4diagnosis():
    """ Distribution of 223 Psychiatric Patients by Type of Psychiatric \
    Disorder and Diagnosis Method.

    This dataset shows the distribution of 223 psychiatric patients by
    diagnosis category and by the method used to obtain the diagnosis in a 4x4
    squared table. The first method named "Clinical Diagnosis" (also known as
    "Facility Diagnosis") is used in a service facility (e.g. public hospital,
    or a community unit) and does not rely on a rigorous application of
    research criteria. The second method known as "Research Diagnosis" is based
    on a strict application of research criteria. Column 1 contains the
    diagnosis categories into which patients are classified with Method 1.
    The first row on the other hand, shows categories into which patients are
    classified with Method 2.

    Returns
    -------
    pandas.DataFrame

        Method
            The method used for the diagnosis.
        Diagnosis
            The category of the diagnosis.
    """
    diagnosis = ["Schizophrenia", "Bipolar Disorder", "Depression", "Other"]
    data = [[40, 4, 4, 17], [6, 25, 2, 13], [4, 1, 21, 12], [15, 5, 9, 45]]
    return pd.DataFrame(data, index=diagnosis, columns=diagnosis)


def distrib_6raters():
    """ Distribution of 6 psychiatrists by Subject/patient and diagnosis \
    category.

    This dataset :cite:p:`Fle71` summarizes the ratings assigned by 6
    psychiatrists classifying 15 patients into one of five categories named
    "Depression", "Personal Disorder", "Schizophrenia", "Neurosis" and "Other.

    Returns
    -------
    pandas.DataFrame

        Units
            This variable represents the unit number.
        Depression
            The number of raters assigned the subject to the depression category
        Personality Disorder
            The number of raters assigned the subject to the personality
            disorder category
        Schizophrenia
            The number of raters assigned the subject to the schizophrenia
            category
        Neurosis
            The number of raters assigned the subject to the neurosis category
        Other
            The number of raters assigned the subject to the other category
    """
    units = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], name="Units")
    data = {
        "Depression": [0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 1, 1, 0, 1, 0],
        "Personality Disorder": [0, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 1, 3, 0, 2],
        "Schizophrenia": [0, 0, 4, 0, 0, 4, 4, 3, 0, 0, 0, 0, 3, 0, 0],
        "Neurosis": [6, 0, 0, 0, 3, 0, 0, 1, 4, 0, 5, 4, 0, 5, 3],
        "Other": [0, 3, 1, 6, 0, 0, 2, 0, 0, 6, 0, 0, 0, 0, 1],
    }
    return pd.DataFrame(data, index=units)
